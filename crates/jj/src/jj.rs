//! Jujutsu (jj) VCS backend.
//!
//! **Reads** (log graph, metadata) go through `jj-lib` directly:
//! [`Workspace::load`] → revset eval → [`TopoGroupedGraphIterator`]. This
//! gives us `GraphEdgeType::{Direct, Indirect, Missing}` as ground truth
//! (the CLI template can't distinguish a direct parent from an ancestor
//! reached through commits elided by the revset), plus topological grouping
//! that keeps branches visually bunched.
//!
//! **Writes** (rebase, bookmark move, abandon, …) still shell out to the
//! `jj` CLI. Porting those to `jj-lib` requires the full transaction
//! machinery — `snapshot_working_copy` (tokio-async FS walk), colocated
//! `git::import_head`/`export_refs`, `workspace.check_out` — roughly 400
//! lines of subtle state handling where a bug means the user loses
//! uncommitted edits. The CLI does this correctly already; subprocess
//! overhead on a user-gesture mutation is ~5ms.
//!
//! Caveat: `jj-lib` pins the on-disk op-log format. If the user's `jj` CLI
//! is upgraded to an incompatible version, reads here will fail until Zed
//! is rebuilt against the matching `jj-lib`.

use anyhow::{Context as _, Result, anyhow, bail};
use collections::HashMap;
use gpui::SharedString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use jj_cli::{
    cli_util::default_ignored_remote_name,
    config::{ConfigEnv, config_from_environment, default_config_layers},
    revset_util,
    ui::Ui,
};
use jj_lib::{
    backend::CommitId,
    config::ConfigNamePathBuf,
    git::REMOTE_NAME_FOR_LOCAL_GIT_REPO,
    graph::{GraphEdgeType, TopoGroupedGraphIterator},
    object_id::ObjectId as _,
    op_heads_store,
    operation::Operation,
    repo::{ReadonlyRepo, Repo, RepoLoaderError, StoreFactories},
    repo_path::RepoPathUiConverter,
    revset::{
        self, Revset, RevsetAliasesMap, RevsetDiagnostics, RevsetExtensions, RevsetParseContext,
        RevsetWorkspaceContext, SymbolResolver, SymbolResolverExtension,
    },
    settings::UserSettings,
    workspace::{self, DefaultWorkspaceLoaderFactory, Workspace, WorkspaceLoaderFactory},
};

use util::command::new_command;

/// A single revision in the jj log.
#[derive(Debug, Clone)]
pub struct JjRevision {
    /// Short change ID (stable across rewrites).
    pub change_id: SharedString,
    /// Full commit ID (unstable — changes on rewrite).
    pub commit_id: SharedString,
    /// Short commit ID prefix for display.
    pub short_commit_id: SharedString,
    /// First line of the commit description.
    pub description: SharedString,
    pub author_name: SharedString,
    pub author_email: SharedString,
    /// Unix timestamp (seconds).
    pub timestamp: i64,
    /// True if this is `@` (the working-copy commit).
    pub is_working_copy: bool,
    /// True if in the immutable set (can't be rewritten without --ignore-immutable).
    pub is_immutable: bool,
    pub has_conflict: bool,
    pub is_empty: bool,
    /// Bookmarks (local and remote) pointing at this revision.
    ///
    /// Locals come first, then remotes. A synced tracked remote (points at the
    /// same commit as its local) is suppressed — the local chip already tells
    /// you where it is.
    pub bookmarks: Vec<JjBookmark>,
    /// Names of workspaces whose `@` (working copy) is at this revision.
    /// Usually empty or `["default"]`; multiple entries mean several
    /// `jj workspace add`-created worktrees are all editing this commit.
    pub workspaces: Vec<SharedString>,
    /// Full commit IDs of parents.
    pub parent_ids: Vec<SharedString>,
}

/// A bookmark reference attached to a revision.
///
/// Matches what `jj log` shows:
///   - local, synced:   `main`
///   - local, unsynced: `main*`   (tracked remote is somewhere else)
///   - remote:          `main@origin`
#[derive(Debug, Clone)]
pub struct JjBookmark {
    pub name: SharedString,
    /// `None` for a local bookmark, `Some("origin")` for a remote-tracking ref.
    pub remote: Option<SharedString>,
    /// For a local bookmark: true if every tracked remote agrees with it
    /// (no `*`). For a remote bookmark: true if it points at the same commit
    /// as the local with the same name.
    pub synced: bool,
    /// Only meaningful for remote bookmarks: whether the user is tracking it
    /// (`jj bookmark track name@remote`). Untracked remotes still render,
    /// but tracked-and-synced remotes are suppressed entirely.
    pub tracked: bool,
}

impl JjBookmark {
    /// The label as `jj log` would render it: `name`, `name*`, or `name@remote`.
    pub fn display(&self) -> SharedString {
        match &self.remote {
            Some(remote) => format!("{}@{}", self.name, remote).into(),
            None if self.synced => self.name.clone(),
            None => format!("{}*", self.name).into(),
        }
    }

    /// Local bookmarks can be dragged to move them; remotes can't
    /// (you move those with `jj git push`).
    pub fn is_local(&self) -> bool {
        self.remote.is_none()
    }
}

/// One row of the laid-out log graph.
///
/// Each row carries enough layout information to render its own horizontal
/// slice of the graph independently — incoming edges from above, passthrough
/// lanes, and outgoing edges below. This lets the UI use a `uniform_list`
/// where every row is a self-contained drag/drop target without needing a
/// shared overlay canvas.
#[derive(Debug, Clone)]
pub struct JjLogEntry {
    pub revision: JjRevision,
    /// Which vertical lane this commit's node sits in (0 = leftmost).
    pub lane: usize,
    /// Edges departing this row toward parents (going down).
    pub edges: Vec<JjGraphEdge>,
    /// Lanes from which an edge arrives at this commit from the row above.
    /// Includes `lane` if this commit has a child directly above it in the
    /// same lane. Empty for head commits (no children in the revset).
    pub incoming_lanes: Vec<usize>,
    /// Lanes that pass straight through this row without touching this
    /// commit — edges going from some ancestor's child to that ancestor,
    /// where the ancestor is below us.
    pub passthrough_lanes: Vec<usize>,
}

/// A connection from a commit node to one of its parents.
///
/// `from_lane` is the lane of the child (this row); `to_lane` is where the
/// edge travels to for the *next* row. If the edge is `indirect`, there are
/// elided commits between child and target — draw it dashed. If the target
/// is outside the query result entirely (`GraphEdgeType::Missing`), the edge
/// trails off.
#[derive(Debug, Clone)]
pub struct JjGraphEdge {
    pub from_lane: usize,
    pub to_lane: usize,
    /// Full commit ID of the target.
    pub to_commit: SharedString,
    /// True if there are elided commits between here and the target, or the
    /// target is missing from the revset entirely.
    pub indirect: bool,
}

/// A handle for running jj operations against a specific workspace.
///
/// Stateless — each operation loads the workspace fresh. This means we never
/// display stale operation-log state even if the user runs `jj` in a terminal
/// alongside us, at the cost of ~5-10ms per refresh on a typical repo.
#[derive(Debug, Clone)]
pub struct JjRepository {
    workspace_root: PathBuf,
}

impl JjRepository {
    /// Walk up from `path` looking for a `.jj/` directory. Returns the
    /// workspace root (the directory containing `.jj/`) if found.
    pub fn discover(path: &Path) -> Option<Self> {
        let mut current = Some(path);
        while let Some(dir) = current {
            if dir.join(".jj").is_dir() {
                return Some(Self {
                    workspace_root: dir.to_path_buf(),
                });
            }
            current = dir.parent();
        }
        None
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    /// Evaluate `revset` against the repository and return laid-out graph rows.
    ///
    /// Uses jj-lib's [`TopoGroupedGraphIterator`], which keeps related
    /// branches grouped in the output order (so they don't interleave) and
    /// returns [`GraphEdge`]s with accurate direct/indirect/missing
    /// classification — something the CLI template can't express.
    pub async fn log(&self, revset: &str) -> Result<Vec<JjLogEntry>> {
        let root = self.workspace_root.clone();
        let revset = revset.to_string();
        smol::unblock(move || log_sync(&root, &revset)).await
    }

    /// Read `revsets.log` from the user's jj config. Falls back to jj's
    /// compiled-in default on any error.
    pub async fn default_revset(&self) -> String {
        let root = self.workspace_root.clone();
        smol::unblock(move || {
            read_config(&root)
                .ok()
                .and_then(|(s, _)| s.get_string("revsets.log").ok())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| {
                    "present(@) | ancestors(immutable_heads().., 2) | trunk()".to_owned()
                })
        })
        .await
    }

    // --- Mutations (CLI) -----------------------------------------------------
    //
    // These stay CLI-backed. The jj-lib transaction path requires
    // snapshot_working_copy (tokio-async), git import/export for colocated
    // repos, and workspace.check_out after every mutation. The CLI handles
    // all of that correctly; a subprocess on user click is fine.

    /// `jj rebase -r <source> -d <destination>`
    ///
    /// Moves a single revision onto a new parent. Descendants are
    /// automatically rebased to follow.
    pub async fn rebase_revision(&self, source_change: &str, dest_change: &str) -> Result<()> {
        self.run_mutation(&["rebase", "-r", source_change, "-d", dest_change])
            .await
    }

    /// `jj rebase -s <source> -d <destination>`
    ///
    /// Moves `source` **and all its descendants** onto `destination`. This is
    /// what you usually want when dragging a commit around — the subtree
    /// travels with it. (`-r` picks off just the one commit and reparents its
    /// children onto its old parents.)
    pub async fn rebase_branch(&self, source_change: &str, dest_change: &str) -> Result<()> {
        self.run_mutation(&["rebase", "-s", source_change, "-d", dest_change])
            .await
    }

    /// `jj bookmark set <name> -r <revision>`
    pub async fn move_bookmark(&self, bookmark: &str, to_change: &str) -> Result<()> {
        self.run_mutation(&[
            "bookmark",
            "set",
            bookmark,
            "-r",
            to_change,
            "--allow-backwards",
        ])
        .await
    }

    /// `jj new <parent>` — create a new empty commit on top of the given revision.
    pub async fn new_revision(&self, parent_change: &str) -> Result<()> {
        self.run_mutation(&["new", parent_change]).await
    }

    /// `jj edit <revision>` — make the given revision the working-copy commit.
    pub async fn edit_revision(&self, change: &str) -> Result<()> {
        self.run_mutation(&["edit", change]).await
    }

    /// `jj abandon <revision>` — drop the revision; children are rebased onto its parents.
    pub async fn abandon_revision(&self, change: &str) -> Result<()> {
        self.run_mutation(&["abandon", change]).await
    }

    /// `jj describe -r <revision> -m <message>`
    pub async fn describe_revision(&self, change: &str, message: &str) -> Result<()> {
        self.run_mutation(&["describe", "-r", change, "-m", message])
            .await
    }

    /// `jj squash --from <source> --into <destination>`
    ///
    /// Moves all changes from source into destination, abandoning source.
    pub async fn squash_into(&self, source_change: &str, dest_change: &str) -> Result<()> {
        self.run_mutation(&["squash", "--from", source_change, "--into", dest_change])
            .await
    }

    /// `jj undo` — undo the last operation.
    pub async fn undo(&self) -> Result<()> {
        self.run_mutation(&["undo"]).await
    }

    async fn run_mutation(&self, args: &[&str]) -> Result<()> {
        let output = new_command("jj")
            .current_dir(&self.workspace_root)
            .arg("--no-pager")
            .args(args)
            .output()
            .await
            .with_context(|| format!("running jj {}", args.join(" ")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("jj {} failed: {}", args.join(" "), stderr.trim());
        }
        Ok(())
    }
}

// --- jj-lib: log query -------------------------------------------------------

/// A loaded jj workspace at its head operation.
///
/// Borrows `settings`/`aliases_map`/etc. are threaded through for revset
/// parsing; `repo` and `wc_id` come from `load_at_head`. Held only for the
/// duration of one `log_sync` call.
struct Session {
    workspace: Workspace,
    settings: UserSettings,
    aliases_map: RevsetAliasesMap,
    extensions: RevsetExtensions,
    path_converter: RepoPathUiConverter,
    repo: Arc<ReadonlyRepo>,
    wc_id: CommitId,
}

impl Session {
    fn load(workspace_root: &Path) -> Result<Self> {
        let (settings, aliases_map) = read_config(workspace_root)?;

        let factory = DefaultWorkspaceLoaderFactory;
        let loader = factory.create(workspace_root).context("locate workspace")?;
        let workspace = loader
            .load(
                &settings,
                &StoreFactories::default(),
                &workspace::default_working_copy_factories(),
            )
            .context("load workspace")?;

        let repo = load_at_head(&workspace)?;

        let wc_id = repo
            .view()
            .get_wc_commit_id(workspace.workspace_name())
            .ok_or_else(|| anyhow!("no working-copy commit for this workspace"))?
            .clone();

        let path_converter = RepoPathUiConverter::Fs {
            cwd: workspace.workspace_root().to_owned(),
            base: workspace.workspace_root().to_owned(),
        };

        Ok(Self {
            workspace,
            settings,
            aliases_map,
            extensions: RevsetExtensions::default(),
            path_converter,
            repo,
            wc_id,
        })
    }

    /// Parse and evaluate a revset expression against this repo.
    ///
    /// The returned [`Revset`] borrows from `self.repo`; the caller must keep
    /// the [`Session`] alive while iterating.
    fn evaluate_revset<'a>(&'a self, revset_str: &str) -> Result<Box<dyn Revset + 'a>> {
        let parse_ctx = self.parse_context();
        let mut diagnostics = RevsetDiagnostics::new();
        let expr = revset::parse(&mut diagnostics, revset_str, &parse_ctx)
            .with_context(|| format!("parse revset `{revset_str}`"))?;
        let expr = revset::optimize(expr);

        let resolver = SymbolResolver::new(
            self.repo.as_ref(),
            &([] as [Box<dyn SymbolResolverExtension>; 0]),
        );
        let resolved = expr
            .resolve_user_expression(self.repo.as_ref(), &resolver)
            .with_context(|| format!("resolve revset `{revset_str}`"))?;

        resolved
            .evaluate(self.repo.as_ref())
            .with_context(|| format!("evaluate revset `{revset_str}`"))
    }

    /// Build a fast `CommitId → bool` check for `immutable_heads()::`.
    ///
    /// Evaluated once per log query; the returned closure is cheap
    /// (index bitmap lookup).
    fn immutable_containing_fn<'a>(
        &'a self,
    ) -> Result<Box<dyn Fn(&CommitId) -> Result<bool> + 'a>> {
        let parse_ctx = self.parse_context();
        let mut diagnostics = RevsetDiagnostics::new();
        let heads = revset_util::parse_immutable_heads_expression(&mut diagnostics, &parse_ctx)
            .context("parse immutable_heads()")?;
        let immutable = heads.ancestors();

        let resolver = SymbolResolver::new(
            self.repo.as_ref(),
            &([] as [Box<dyn SymbolResolverExtension>; 0]),
        );
        let resolved = immutable
            .resolve_user_expression(self.repo.as_ref(), &resolver)
            .context("resolve immutable_heads()::")?;
        let revset = resolved
            .evaluate(self.repo.as_ref())
            .context("evaluate immutable_heads()::")?;

        let contains = revset.containing_fn();
        Ok(Box::new(move |id| Ok(contains(id)?)))
    }

    fn parse_context(&self) -> RevsetParseContext<'_> {
        let ws_ctx = RevsetWorkspaceContext {
            path_converter: &self.path_converter,
            workspace_name: self.workspace.workspace_name(),
        };
        RevsetParseContext {
            aliases_map: &self.aliases_map,
            local_variables: std::collections::HashMap::new(),
            user_email: self.settings.user_email(),
            date_pattern_context: chrono::Local::now().into(),
            default_ignored_remote: default_ignored_remote_name(self.repo.store()),
            extensions: &self.extensions,
            workspace: Some(ws_ctx),
            use_glob_by_default: false,
        }
    }
}

/// Load merged jj config (defaults + user `~/.jjconfig.toml` + repo
/// `.jj/repo/config.toml`) and pull the revset alias table from it.
///
/// The alias table is required for the default revset to parse at all:
/// `immutable_heads()`, `trunk()`, etc. are all user-configurable aliases,
/// not builtin functions.
fn read_config(workspace_root: &Path) -> Result<(UserSettings, RevsetAliasesMap)> {
    let mut config_env = ConfigEnv::from_environment();
    let mut raw = config_from_environment(default_config_layers());

    config_env.reload_user_config(&mut raw)?;
    config_env.reset_repo_path(&workspace_root.join(".jj").join("repo"));
    // Ignore repo-config load failures (malformed TOML etc.) rather than
    // killing the whole panel — we'll still have defaults + user config.
    let _ = config_env.reload_repo_config(&Ui::null(), &mut raw);

    let config = config_env.resolve_config(&raw)?;
    let aliases_map = build_aliases_map(&config)?;
    let settings = UserSettings::from_config(config)?;

    Ok((settings, aliases_map))
}

/// Walk all `[revset-aliases]` tables across the config stack (in priority
/// order, so later layers override earlier ones) and collect them into a
/// single [`RevsetAliasesMap`].
fn build_aliases_map(config: &jj_lib::config::StackedConfig) -> Result<RevsetAliasesMap> {
    let table_name = ConfigNamePathBuf::from_iter(["revset-aliases"]);
    let mut map = RevsetAliasesMap::new();
    for layer in config.layers() {
        let Ok(Some(table)) = layer.look_up_table(&table_name) else {
            continue;
        };
        for (decl, item) in table.iter() {
            let Some(value) = item.as_str() else {
                continue;
            };
            map.insert(decl, value)
                .map_err(|e| anyhow!("revset-aliases.{decl}: {e}"))?;
        }
    }
    Ok(map)
}

/// Load the repo at whichever operation is current head.
///
/// If there are concurrent op-heads (user ran `jj` in two terminals
/// simultaneously and neither saw the other yet), merge them first. This
/// is straight from jj-cli's init path.
fn load_at_head(workspace: &Workspace) -> Result<Arc<ReadonlyRepo>> {
    let loader = workspace.repo_loader();
    let op = op_heads_store::resolve_op_heads(
        loader.op_heads_store().as_ref(),
        loader.op_store(),
        |op_heads| {
            let base_repo = loader.load_at(&op_heads[0])?;
            let mut tx = base_repo.start_transaction();
            for other in op_heads.into_iter().skip(1) {
                tx.merge_operation(other)?;
                tx.repo_mut().rebase_descendants()?;
            }
            Ok::<Operation, RepoLoaderError>(
                tx.write("resolve concurrent operations")?
                    .leave_unpublished()
                    .operation()
                    .clone(),
            )
        },
    )?;
    loader.load_at(&op).context("load op head")
}

/// Synchronous body of [`JjRepository::log`]. Runs inside `smol::unblock`.
fn log_sync(workspace_root: &Path, revset_str: &str) -> Result<Vec<JjLogEntry>> {
    let session = Session::load(workspace_root)?;

    let is_immutable = session.immutable_containing_fn()?;

    let bookmark_index = build_bookmark_index(&session.repo);

    // CommitId → workspace names. Typically a single entry (default → @) but
    // `jj workspace add` creates more.
    let mut workspace_index: HashMap<CommitId, Vec<SharedString>> = HashMap::default();
    for (ws_name, commit_id) in session.repo.view().wc_commit_ids() {
        workspace_index
            .entry(commit_id.clone())
            .or_default()
            .push(SharedString::from(ws_name.as_str().to_owned()));
    }

    let revset = session.evaluate_revset(revset_str)?;
    let root_id = session.repo.store().root_commit_id().clone();

    // The `as_id` closure is jj-lib asking "how do I get a hashable identity
    // out of your node type"; for us the node IS the id, so identity.
    let iter = TopoGroupedGraphIterator::new(revset.iter_graph(), |id: &CommitId| id);

    let mut rows: Vec<(JjRevision, Vec<GraphEdgeInput>)> = Vec::new();
    for item in iter {
        let (commit_id, graph_edges) = item?;
        let commit = session.repo.store().get_commit(&commit_id)?;
        let author = commit.author();

        // Convert the jj-lib graph edges into the decoupled form the layout
        // pass consumes. `Missing` is how jj-lib says "target isn't in the
        // revset at all" (typically the root's synthetic parent); `Indirect`
        // means there are real commits between here and the target that the
        // revset filtered out.
        let edges: Vec<GraphEdgeInput> = graph_edges
            .into_iter()
            .filter_map(|e| {
                let missing = e.edge_type == GraphEdgeType::Missing;
                // Skip the phantom missing-edge to the root's nonexistent parent.
                if missing && e.target == root_id {
                    return None;
                }
                Some(GraphEdgeInput {
                    target: e.target.hex().into(),
                    indirect: e.edge_type != GraphEdgeType::Direct,
                    missing,
                })
            })
            .collect();

        let change_hex = commit.change_id().reverse_hex();
        let commit_hex = commit_id.hex();

        let revision = JjRevision {
            change_id: SharedString::from(change_hex[..change_hex.len().min(12)].to_owned()),
            commit_id: commit_hex.clone().into(),
            short_commit_id: SharedString::from(commit_hex[..commit_hex.len().min(12)].to_owned()),
            description: commit
                .description()
                .lines()
                .next()
                .unwrap_or_default()
                .to_owned()
                .into(),
            author_name: author.name.clone().into(),
            author_email: author.email.clone().into(),
            // jj stores millis since epoch; the panel expects seconds.
            timestamp: author.timestamp.timestamp.0 / 1000,
            is_working_copy: commit_id == session.wc_id,
            is_immutable: is_immutable(&commit_id)?,
            has_conflict: commit.has_conflict(),
            is_empty: commit.is_empty(session.repo.as_ref())?,
            bookmarks: bookmark_index.get(&commit_id).cloned().unwrap_or_default(),
            workspaces: workspace_index
                .get(&commit_id)
                .cloned()
                .unwrap_or_default(),
            parent_ids: commit.parent_ids().iter().map(|p| p.hex().into()).collect(),
        };

        rows.push((revision, edges));
    }

    Ok(layout_graph(rows))
}

/// Build a `CommitId → [JjBookmark]` index covering both local and remote refs.
///
/// Semantics follow `jj log`:
///   - Local bookmarks always appear. `synced` is false (→ `*` suffix) if any
///     tracked remote points elsewhere. The `git` pseudo-remote (colocated
///     repos mirror local bookmarks into `.git/refs`) is ignored for this.
///   - Remote bookmarks appear only where they *differ* from the local:
///     untracked remotes always show; tracked remotes show only when the
///     target diverged (tracked+synced would be noise on the same row as the
///     local chip).
///
/// This is a direct port of GG's `build_ref_index`.
fn build_bookmark_index(repo: &ReadonlyRepo) -> HashMap<CommitId, Vec<JjBookmark>> {
    let mut index: HashMap<CommitId, Vec<JjBookmark>> = HashMap::default();

    let mut insert = |ids: &mut dyn Iterator<Item = &CommitId>, bm: JjBookmark| {
        for id in ids {
            index.entry(id.clone()).or_default().push(bm.clone());
        }
    };

    for (name, targets) in repo.view().bookmarks() {
        let local = targets.local_target;
        let remotes = targets.remote_refs;

        if local.is_present() {
            // Out of sync if any tracked real remote disagrees.
            let synced = remotes.iter().all(|&(remote_name, remote_ref)| {
                remote_name == REMOTE_NAME_FOR_LOCAL_GIT_REPO
                    || !remote_ref.is_tracked()
                    || remote_ref.target == *local
            });
            insert(
                &mut local.added_ids(),
                JjBookmark {
                    name: name.as_str().to_owned().into(),
                    remote: None,
                    synced,
                    tracked: true, // meaningless for locals, but keep it truthy
                },
            );
        }

        for &(remote_name, remote_ref) in &remotes {
            if remote_name == REMOTE_NAME_FOR_LOCAL_GIT_REPO {
                continue;
            }
            let synced = remote_ref.target == *local;
            let tracked = remote_ref.is_tracked();
            // Suppress tracked+synced: the local chip already covers it.
            // Untracked remotes always show (they're independent refs the user
            // hasn't opted into tracking). Local-absent always shows (no local
            // chip to cover it).
            if tracked && synced && local.is_present() {
                continue;
            }
            insert(
                &mut remote_ref.target.added_ids(),
                JjBookmark {
                    name: name.as_str().to_owned().into(),
                    remote: Some(remote_name.as_str().to_owned().into()),
                    synced,
                    tracked,
                },
            );
        }
    }

    index
}

// --- Lane layout -------------------------------------------------------------

/// Graph-edge input to the layout pass, decoupled from `jj_lib::graph` types
/// so tests can build them without constructing real `CommitId`s.
#[derive(Debug, Clone)]
struct GraphEdgeInput {
    /// Hex-encoded commit id of the edge's target.
    target: SharedString,
    /// The edge passes through elided commits (dashed line).
    indirect: bool,
    /// The target is outside the revset entirely — don't reserve a lane for
    /// it, just let the edge trail off to a phantom column.
    missing: bool,
}

/// Assign each revision to a vertical lane and compute the outgoing edges.
///
/// Classic "stem" layout: we maintain a list of open lanes, each waiting for
/// a particular target commit. When a commit arrives, it claims the leftmost
/// lane that was waiting for it (other lanes waiting for the same commit
/// collapse into it — merges). Its own edges then reserve lanes for the rows
/// below.
///
/// `TopoGroupedGraphIterator` already hands us rows children-before-parents
/// with related branches clustered together, which is exactly the order this
/// algorithm wants.
fn layout_graph(rows: Vec<(JjRevision, Vec<GraphEdgeInput>)>) -> Vec<JjLogEntry> {
    // Each slot holds the target commit id it's waiting for, or None if free.
    let mut lanes: Vec<Option<SharedString>> = Vec::new();
    let mut entries = Vec::with_capacity(rows.len());

    for (rev, input_edges) in rows {
        // Find all lanes waiting for this commit.
        let incoming_lanes: Vec<usize> = lanes
            .iter()
            .enumerate()
            .filter_map(|(i, waiting)| {
                waiting
                    .as_ref()
                    .filter(|w| **w == rev.commit_id)
                    .map(|_| i)
            })
            .collect();

        // Lanes that are active right now but NOT arriving here pass straight
        // through this row — they're edges going to some commit further down.
        let passthrough_lanes: Vec<usize> = lanes
            .iter()
            .enumerate()
            .filter_map(|(i, waiting)| match waiting {
                Some(w) if *w != rev.commit_id => Some(i),
                _ => None,
            })
            .collect();

        // Place this commit on the leftmost incoming lane, or a fresh lane if none.
        let my_lane = if let Some(first) = incoming_lanes.first().copied() {
            // Clear all incoming lanes (they all merge here).
            for i in incoming_lanes.iter().skip(1) {
                lanes[*i] = None;
            }
            lanes[first] = None;
            first
        } else {
            // No lane was waiting — this is a head. Grab the first free slot.
            match lanes.iter().position(Option::is_none) {
                Some(i) => i,
                None => {
                    lanes.push(None);
                    lanes.len() - 1
                }
            }
        };

        // Reserve lanes for outgoing edges. First edge stays in our lane;
        // additional edges spawn new lanes (or join existing ones waiting for
        // the same target — shared ancestor).
        let mut edges = Vec::with_capacity(input_edges.len());
        for (edge_idx, input) in input_edges.into_iter().enumerate() {
            let to_lane = if edge_idx == 0 {
                // First edge continues in our lane. Missing edges don't
                // reserve — the lane is free below us.
                if !input.missing {
                    if lanes.len() <= my_lane {
                        lanes.resize(my_lane + 1, None);
                    }
                    lanes[my_lane] = Some(input.target.clone());
                }
                my_lane
            } else if input.missing {
                // Target outside the revset — phantom lane one past the
                // rightmost real one; don't actually reserve it.
                lanes.len()
            } else if let Some(existing) =
                lanes.iter().position(|w| w.as_ref() == Some(&input.target))
            {
                // Another lane is already headed to this same target. Join it
                // instead of spawning a parallel line that'll merge anyway.
                existing
            } else {
                // Spawn a new lane.
                match lanes.iter().position(Option::is_none) {
                    Some(i) => {
                        lanes[i] = Some(input.target.clone());
                        i
                    }
                    None => {
                        lanes.push(Some(input.target.clone()));
                        lanes.len() - 1
                    }
                }
            };

            edges.push(JjGraphEdge {
                from_lane: my_lane,
                to_lane,
                to_commit: input.target,
                indirect: input.indirect,
            });
        }

        // Trim trailing empty lanes so width doesn't grow unboundedly.
        while lanes.last() == Some(&None) {
            lanes.pop();
        }

        entries.push(JjLogEntry {
            revision: rev,
            lane: my_lane,
            edges,
            incoming_lanes,
            passthrough_lanes,
        });
    }

    entries
}

/// Returns a map from commit_id → row index for fast edge resolution during rendering.
pub fn build_commit_index(entries: &[JjLogEntry]) -> HashMap<SharedString, usize> {
    entries
        .iter()
        .enumerate()
        .map(|(i, e)| (e.revision.commit_id.clone(), i))
        .collect()
}

/// Maximum lane index used across all entries. Used to size the graph canvas.
pub fn max_lanes(entries: &[JjLogEntry]) -> usize {
    entries
        .iter()
        .flat_map(|e| std::iter::once(e.lane).chain(e.edges.iter().map(|edge| edge.to_lane)))
        .max()
        .map(|m| m + 1)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rev(id: &str) -> JjRevision {
        JjRevision {
            change_id: id.to_string().into(),
            commit_id: id.to_string().into(),
            short_commit_id: id.to_string().into(),
            description: "".into(),
            author_name: "".into(),
            author_email: "".into(),
            timestamp: 0,
            is_working_copy: false,
            is_immutable: false,
            has_conflict: false,
            is_empty: false,
            bookmarks: vec![],
            workspaces: vec![],
            parent_ids: vec![],
        }
    }

    fn direct(target: &str) -> GraphEdgeInput {
        GraphEdgeInput {
            target: target.to_string().into(),
            indirect: false,
            missing: false,
        }
    }

    #[test]
    fn test_layout_linear() {
        // c → b → a (c is newest, a is oldest)
        let rows = vec![
            (rev("c"), vec![direct("b")]),
            (rev("b"), vec![direct("a")]),
            (rev("a"), vec![]),
        ];
        let entries = layout_graph(rows);
        assert_eq!(entries.len(), 3);
        for e in &entries {
            assert_eq!(e.lane, 0);
        }
        // a has one incoming lane (from b in lane 0).
        assert_eq!(entries[2].incoming_lanes, vec![0]);
    }

    #[test]
    fn test_layout_merge() {
        // Merge: d has two parents b and c, which both descend from a.
        //   d
        //  / \
        // b   c
        //  \ /
        //   a
        let rows = vec![
            (rev("d"), vec![direct("b"), direct("c")]),
            (rev("b"), vec![direct("a")]),
            (rev("c"), vec![direct("a")]),
            (rev("a"), vec![]),
        ];
        let entries = layout_graph(rows);
        assert_eq!(entries[0].lane, 0); // d
        assert_eq!(entries[0].edges.len(), 2);
        // b in lane 0 (first edge of d stays in-lane)
        assert_eq!(entries[1].lane, 0);
        // c in lane 1 (second edge spawned a new lane)
        assert_eq!(entries[2].lane, 1);
        // a merges both back — lands in leftmost incoming lane
        assert_eq!(entries[3].lane, 0);
        assert_eq!(entries[3].incoming_lanes, vec![0, 1]);
    }

    #[test]
    fn test_layout_indirect_edge() {
        // c →(indirect) a, with b present but not in c's path.
        // c's edge to a should be dashed but still reserve lane 0.
        let rows = vec![
            (
                rev("c"),
                vec![GraphEdgeInput {
                    target: "a".to_string().into(),
                    indirect: true,
                    missing: false,
                }],
            ),
            (rev("a"), vec![]),
        ];
        let entries = layout_graph(rows);
        assert_eq!(entries[0].edges[0].indirect, true);
        assert_eq!(entries[0].edges[0].to_lane, 0);
        // a should have received the incoming lane from c
        assert_eq!(entries[1].incoming_lanes, vec![0]);
    }

    #[test]
    fn test_layout_missing_edge_no_lane() {
        // c → b(missing). Missing edges don't reserve a lane — c is in lane 0
        // but lane 0 stays free for the next head.
        let rows = vec![
            (
                rev("c"),
                vec![GraphEdgeInput {
                    target: "b".to_string().into(),
                    indirect: true,
                    missing: true,
                }],
            ),
            (rev("d"), vec![]), // unrelated head should reuse lane 0
        ];
        let entries = layout_graph(rows);
        assert_eq!(entries[0].lane, 0);
        assert_eq!(entries[1].lane, 0); // d reused lane 0 — missing edge didn't reserve it
        assert_eq!(entries[1].incoming_lanes, Vec::<usize>::new());
    }
}
