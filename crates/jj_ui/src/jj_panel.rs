use anyhow::{Context as _, Result};
use collections::HashMap;
use gpui::{
    Action, App, AppContext as _, AsyncWindowContext, Bounds, ClickEvent, Corner, DismissEvent,
    Entity, EventEmitter, FocusHandle, Focusable, Hsla, MouseButton, MouseDownEvent, PathBuilder,
    Pixels, Point, SharedString, Subscription, Task, UniformListScrollHandle, WeakEntity, Window,
    anchored, canvas, deferred, div, point, px, uniform_list,
};
use jj::{JjBookmark, JjGraphEdge, JjLogEntry, JjRepository};
use project::{Project, Worktree};
use settings::Settings;
use std::{ops::Range, sync::Arc, time::Duration};
use theme::ThemeSettings;
use ui::{ContextMenu, Tooltip, prelude::*};
use workspace::{
    Workspace,
    dock::{DockPosition, Panel, PanelEvent},
};

use crate::{Refresh, ToggleFocus};

/// How often we poll `.jj/repo/op_heads/heads/` for external changes.
const OP_HEADS_POLL_INTERVAL: Duration = Duration::from_secs(1);

// ─── Layout constants ───────────────────────────────────────────────────────
// Tuned to match Zed's git_graph density but slightly wider nodes so the
// working-copy marker (filled @) reads clearly at small sizes.

const LANE_WIDTH: Pixels = px(14.0);
const GRAPH_LEFT_PAD: Pixels = px(10.0);
const GRAPH_RIGHT_PAD: Pixels = px(8.0);
const NODE_RADIUS: Pixels = px(4.0);
const LINE_WIDTH: Pixels = px(1.5);
const ROW_HEIGHT: Pixels = px(24.0);
const DEFAULT_PANEL_WIDTH: Pixels = px(360.0);

const JJ_PANEL_KEY: &str = "JjPanel";

// ─── Drag payloads ──────────────────────────────────────────────────────────

/// Payload carried when dragging a revision node. Dropping on another row
/// performs `jj rebase -r <change_id> -d <target>`.
#[derive(Clone, Debug)]
struct DraggedJjRevision {
    change_id: SharedString,
    description: SharedString,
}

/// Payload carried when dragging a bookmark chip. Dropping on a row performs
/// `jj bookmark set <name> -r <target>`.
#[derive(Clone, Debug)]
struct DraggedJjBookmark {
    name: SharedString,
    from_change: SharedString,
}

/// Floating preview shown under the cursor while dragging a revision.
struct DraggedJjRevisionView {
    change_id: SharedString,
    description: SharedString,
    click_offset: Point<Pixels>,
}

impl Render for DraggedJjRevisionView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let ui_font = ThemeSettings::get_global(cx).ui_font.clone();
        h_flex()
            .font(ui_font)
            .pl(self.click_offset.x + px(8.))
            .pt(self.click_offset.y + px(8.))
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .py_1()
                    .px_2()
                    .rounded_md()
                    .bg(cx.theme().colors().background)
                    .border_1()
                    .border_color(cx.theme().colors().border)
                    .shadow_md()
                    .child(
                        Label::new(self.change_id.clone())
                            .size(LabelSize::Small)
                            .color(Color::Accent),
                    )
                    .child(Label::new(if self.description.is_empty() {
                        SharedString::from("(no description)")
                    } else {
                        self.description.clone()
                    })),
            )
    }
}

/// Floating preview shown under the cursor while dragging a bookmark.
struct DraggedJjBookmarkView {
    name: SharedString,
    click_offset: Point<Pixels>,
}

impl Render for DraggedJjBookmarkView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let ui_font = ThemeSettings::get_global(cx).ui_font.clone();
        h_flex()
            .font(ui_font)
            .pl(self.click_offset.x + px(8.))
            .pt(self.click_offset.y + px(8.))
            .child(
                h_flex()
                    .gap_1()
                    .items_center()
                    .py_0p5()
                    .px_1p5()
                    .rounded_sm()
                    .bg(cx.theme().status().info_background)
                    .border_1()
                    .border_color(cx.theme().status().info_border)
                    .shadow_md()
                    .child(Icon::new(IconName::GitBranch).size(IconSize::XSmall))
                    .child(Label::new(self.name.clone()).size(LabelSize::Small)),
            )
    }
}

// ─── Panel ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
enum PanelState {
    /// No `.jj/` directory found in any project worktree.
    NotAJjRepo,
    /// Initial load running.
    Loading,
    /// Log loaded and laid out.
    Loaded {
        entries: Arc<[JjLogEntry]>,
        /// Width in lanes of the widest row.
        max_lanes: usize,
    },
    /// Load failed.
    Error(SharedString),
}

pub struct JjPanel {
    focus_handle: FocusHandle,
    workspace: WeakEntity<Workspace>,
    project: Entity<Project>,

    repository: Option<JjRepository>,
    state: PanelState,
    /// Revset fed to `jj log`. Defaults to the user's `revsets.log` config.
    revset: String,

    selected_ix: Option<usize>,
    /// Row currently being hovered by a drag. Shown with a drop-target highlight.
    drag_target_ix: Option<usize>,

    context_menu: Option<(Entity<ContextMenu>, Point<Pixels>, Subscription)>,

    scroll_handle: UniformListScrollHandle,
    width: Option<Pixels>,

    /// Prevents stale log results from clobbering newer ones when refreshes overlap.
    refresh_epoch: usize,
    pending_refresh: Option<Task<()>>,
    pending_mutation: Option<Task<()>>,

    /// Background loop polling `.jj/repo/op_heads/heads/` for external changes.
    /// Dropped (cancelled) when the repository changes or the panel is dropped.
    _op_watcher: Option<Task<()>>,

    _subscriptions: Vec<Subscription>,
}

impl JjPanel {
    /// Called by Zed's panel loader. Mirrors `GitPanel::load`.
    pub async fn load(
        workspace: WeakEntity<Workspace>,
        mut cx: AsyncWindowContext,
    ) -> Result<Entity<Self>> {
        workspace.update_in(&mut cx, |workspace, window, cx| {
            Self::new(workspace, window, cx)
        })
    }

    fn new(
        workspace: &mut Workspace,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> Entity<Self> {
        let project = workspace.project().clone();
        let weak_workspace = workspace.weak_handle();

        cx.new(|cx| {
            let focus_handle = cx.focus_handle();

            // Re-discover the repo whenever worktrees change (e.g. user opens a
            // different folder).
            let mut subs = Vec::new();
            subs.push(
                cx.observe_in(&project, window, |this: &mut Self, _, _, cx| {
                    this.discover_repository(cx);
                    this.refresh(cx);
                }),
            );

            let mut this = Self {
                focus_handle,
                workspace: weak_workspace,
                project,
                repository: None,
                state: PanelState::NotAJjRepo,
                revset: String::new(),
                selected_ix: None,
                drag_target_ix: None,
                context_menu: None,
                scroll_handle: UniformListScrollHandle::new(),
                width: None,
                refresh_epoch: 0,
                pending_refresh: None,
                pending_mutation: None,
                _op_watcher: None,
                _subscriptions: subs,
            };

            this.discover_repository(cx);
            this.refresh(cx);
            this.start_op_watcher(cx);
            this
        })
    }

    /// Walk project worktrees looking for a `.jj/` root. We take the first match.
    fn discover_repository(&mut self, cx: &mut Context<Self>) {
        let new_repo = self
            .project
            .read(cx)
            .visible_worktrees(cx)
            .find_map(|worktree: Entity<Worktree>| {
                let abs_path = worktree.read(cx).abs_path();
                JjRepository::discover(&abs_path)
            });

        let changed = match (&self.repository, &new_repo) {
            (None, None) => false,
            (Some(a), Some(b)) => a.workspace_root() != b.workspace_root(),
            _ => true,
        };

        self.repository = new_repo;

        if changed {
            self.selected_ix = None;
            self.revset.clear();
            self.start_op_watcher(cx);
        }
    }

    /// Spawn a background loop that polls `.jj/repo/op_heads/heads/` for
    /// changes every [`OP_HEADS_POLL_INTERVAL`]. When the directory's mtime
    /// changes (a new jj operation was committed — by the user's terminal,
    /// another tool, or our own mutations), we trigger a panel refresh.
    ///
    /// This is the simplest reliable cross-platform approach. Proper
    /// `inotify`/`kqueue` watching would be more efficient but Zed's fs
    /// watcher doesn't cover `.jj/` (it's in `.gitignore` for colocated
    /// repos).
    fn start_op_watcher(&mut self, cx: &mut Context<Self>) {
        // Drop the old watcher (cancels the task).
        self._op_watcher = None;

        let Some(repo) = self.repository.as_ref() else {
            return;
        };
        let op_heads_dir = repo.workspace_root().join(".jj/repo/op_heads/heads");

        self._op_watcher = Some(cx.spawn(async move |this, cx| {
            let mut last_mtime = cx
                .background_spawn({
                    let d = op_heads_dir.clone();
                    async move { std::fs::metadata(&d).and_then(|m| m.modified()).ok() }
                })
                .await;

            loop {
                cx.background_executor().timer(OP_HEADS_POLL_INTERVAL).await;

                let current_mtime = cx
                    .background_spawn({
                        let d = op_heads_dir.clone();
                        async move { std::fs::metadata(&d).and_then(|m| m.modified()).ok() }
                    })
                    .await;

                if current_mtime != last_mtime {
                    last_mtime = current_mtime;
                    let ok = this
                        .update(cx, |this, cx| {
                            // Skip if a mutation is already in flight — it will
                            // trigger its own refresh when it completes.
                            if this.pending_mutation.is_none() {
                                this.refresh(cx);
                            }
                        });
                    if ok.is_err() {
                        // Entity dropped — stop polling.
                        break;
                    }
                }
            }
        }));
    }

    /// Kick off a background `jj log` and swap the result in when it arrives.
    fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(repo) = self.repository.clone() else {
            self.state = PanelState::NotAJjRepo;
            cx.notify();
            return;
        };

        // First refresh: fetch the user's default revset before loading.
        let need_revset = self.revset.is_empty();
        self.refresh_epoch = self.refresh_epoch.wrapping_add(1);
        let epoch = self.refresh_epoch;

        if matches!(self.state, PanelState::NotAJjRepo | PanelState::Error(_)) {
            self.state = PanelState::Loading;
        }

        let revset = self.revset.clone();
        self.pending_refresh = Some(cx.spawn(async move |this, cx| {
            let revset = if need_revset {
                repo.default_revset().await
            } else {
                revset
            };

            let result = repo.log(&revset).await;

            this.update(cx, |this, cx| {
                if this.refresh_epoch != epoch {
                    // A newer refresh was started; drop this result on the floor.
                    return;
                }
                if need_revset {
                    this.revset = revset;
                }
                match result {
                    Ok(entries) => {
                        // Preserve selection across refreshes by change_id, since
                        // commit_ids churn on every rewrite in jj.
                        let prev_change = this.selected_change_id();
                        let change_index: HashMap<SharedString, usize> = entries
                            .iter()
                            .enumerate()
                            .map(|(i, e)| (e.revision.change_id.clone(), i))
                            .collect();
                        this.selected_ix =
                            prev_change.and_then(|c| change_index.get(&c).copied());

                        let max_lanes = jj::max_lanes(&entries);
                        this.state = PanelState::Loaded {
                            entries: entries.into(),
                            max_lanes,
                        };
                    }
                    Err(err) => {
                        this.state = PanelState::Error(err.to_string().into());
                    }
                }
                cx.notify();
            })
            .ok();
        }));

        cx.notify();
    }

    fn selected_change_id(&self) -> Option<SharedString> {
        match &self.state {
            PanelState::Loaded { entries, .. } => self
                .selected_ix
                .and_then(|ix| entries.get(ix))
                .map(|e| e.revision.change_id.clone()),
            _ => None,
        }
    }

    /// Run a mutation closure against the repo in the background, then refresh.
    /// Errors are surfaced via the workspace toast system.
    fn run_mutation<F, Fut>(&mut self, description: &'static str, f: F, cx: &mut Context<Self>)
    where
        F: FnOnce(JjRepository) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send,
    {
        let Some(repo) = self.repository.clone() else {
            return;
        };
        let workspace = self.workspace.clone();
        self.pending_mutation = Some(cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move { f(repo).await })
                .await
                .with_context(|| description);

            this.update(cx, |this, cx| {
                match result {
                    Ok(()) => {
                        this.refresh(cx);
                    }
                    Err(err) => {
                        log::error!("jj mutation '{description}' failed: {err:#}");
                        workspace
                            .update(cx, |workspace, cx| {
                                workspace.show_error(&err, cx);
                            })
                            .ok();
                        this.refresh(cx);
                    }
                }
            })
            .ok();
        }));
    }

    // ─── Mutation wrappers (called from drag handlers & context menu) ─────

    /// Rebase `source` and its descendants onto `dest` (`jj rebase -s`).
    /// This is the default drag-drop behavior — the subtree travels together.
    fn rebase_branch(&mut self, source: SharedString, dest: SharedString, cx: &mut Context<Self>) {
        if source == dest {
            return;
        }
        self.run_mutation(
            "rebase -s",
            move |repo| async move { repo.rebase_branch(&source, &dest).await },
            cx,
        );
    }

    fn move_bookmark(&mut self, name: SharedString, to: SharedString, cx: &mut Context<Self>) {
        self.run_mutation(
            "move bookmark",
            move |repo| async move { repo.move_bookmark(&name, &to).await },
            cx,
        );
    }

    fn new_child(&mut self, parent: SharedString, cx: &mut Context<Self>) {
        self.run_mutation(
            "new",
            move |repo| async move { repo.new_revision(&parent).await },
            cx,
        );
    }

    fn edit(&mut self, change: SharedString, cx: &mut Context<Self>) {
        self.run_mutation(
            "edit",
            move |repo| async move { repo.edit_revision(&change).await },
            cx,
        );
    }

    fn abandon(&mut self, change: SharedString, cx: &mut Context<Self>) {
        self.run_mutation(
            "abandon",
            move |repo| async move { repo.abandon_revision(&change).await },
            cx,
        );
    }

    fn squash_into(&mut self, source: SharedString, dest: SharedString, cx: &mut Context<Self>) {
        if source == dest {
            return;
        }
        self.run_mutation(
            "squash",
            move |repo| async move { repo.squash_into(&source, &dest).await },
            cx,
        );
    }

    fn undo(&mut self, cx: &mut Context<Self>) {
        self.run_mutation("undo", |repo| async move { repo.undo().await }, cx);
    }

    // ─── Context menu ──────────────────────────────────────────────────────

    fn deploy_context_menu(
        &mut self,
        ix: usize,
        position: Point<Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let PanelState::Loaded { entries, .. } = &self.state else {
            return;
        };
        let Some(entry) = entries.get(ix) else {
            return;
        };
        let change_id = entry.revision.change_id.clone();
        let is_immutable = entry.revision.is_immutable;
        let is_wc = entry.revision.is_working_copy;
        let has_parent = !entry.revision.parent_ids.is_empty();

        let this = cx.entity();

        let context_menu = ContextMenu::build(window, cx, |menu, _, _| {
            menu.context(self.focus_handle.clone())
                .entry("New Child", None, {
                    let this = this.clone();
                    let change = change_id.clone();
                    move |_, cx| {
                        this.update(cx, |this, cx| this.new_child(change.clone(), cx));
                    }
                })
                .when(!is_wc, |menu| {
                    menu.entry("Edit", None, {
                        let this = this.clone();
                        let change = change_id.clone();
                        move |_, cx| {
                            this.update(cx, |this, cx| this.edit(change.clone(), cx));
                        }
                    })
                })
                .when(!is_immutable && has_parent, |menu| {
                    menu.separator()
                        .entry("Squash into Parent", None, {
                            let this = this.clone();
                            let change = change_id.clone();
                            move |_, cx| {
                                // @- is "parent of working copy" in jj revset syntax, but
                                // here we want "parent of this change": `<change>-`.
                                let parent = SharedString::from(format!("{}-", change));
                                this.update(cx, |this, cx| {
                                    this.squash_into(change.clone(), parent, cx)
                                });
                            }
                        })
                        .entry("Abandon", None, {
                            let this = this.clone();
                            let change = change_id.clone();
                            move |_, cx| {
                                this.update(cx, |this, cx| this.abandon(change.clone(), cx));
                            }
                        })
                })
                .separator()
                .entry("Refresh", None, {
                    let this = this.clone();
                    move |_, cx| {
                        this.update(cx, |this, cx| this.refresh(cx));
                    }
                })
                .entry("Undo Last Operation", None, {
                    let this = this.clone();
                    move |_, cx| {
                        this.update(cx, |this, cx| this.undo(cx));
                    }
                })
        });

        self.selected_ix = Some(ix);
        let subscription =
            cx.subscribe_in(&context_menu, window, |this, _, _: &DismissEvent, _, cx| {
                this.context_menu.take();
                cx.notify();
            });
        self.context_menu = Some((context_menu, position, subscription));
        cx.notify();
    }

    // ─── Rendering ─────────────────────────────────────────────────────────

    /// Render one row of the log. The row is a horizontal flex with:
    ///   - a fixed-width canvas on the left painting this row's slice of the graph
    ///   - the commit description, bookmarks, and metadata on the right
    ///
    /// Each row is both a drag source (for rebase) and a drop target (for both
    /// rebased revisions and moved bookmarks).
    fn render_row(
        &self,
        ix: usize,
        entry: &JjLogEntry,
        graph_width: Pixels,
        window: &Window,
        cx: &Context<Self>,
    ) -> impl IntoElement + use<> {
        let rev = &entry.revision;
        let is_selected = self.selected_ix == Some(ix);
        let is_drag_target = self.drag_target_ix == Some(ix);
        let is_focused = self.focus_handle.is_focused(window);

        let target_change = rev.change_id.clone();
        let target_change_for_rev = target_change.clone();
        let target_change_for_bm = target_change.clone();
        let target_change_for_click = target_change.clone();

        let drag_payload = DraggedJjRevision {
            change_id: rev.change_id.clone(),
            description: rev.description.clone(),
        };

        let description = if rev.description.is_empty() {
            SharedString::from("(no description)")
        } else {
            rev.description.clone()
        };

        let description_color = if rev.description.is_empty() {
            Color::Placeholder
        } else if rev.is_immutable {
            Color::Muted
        } else {
            Color::Default
        };

        let id = ElementId::NamedInteger("jj-row".into(), ix as u64);

        // Graph slice for this row.
        let graph_cell = self.render_graph_cell(entry, graph_width, cx);

        let bookmarks: Vec<JjBookmark> = rev.bookmarks.clone();
        let workspaces = rev.workspaces.clone();
        let change_id_for_bm = rev.change_id.clone();
        // Change IDs are stable across rewrites; that's the one you want to
        // read & reference. Commit ID is still in the tooltip.
        let display_id = rev.change_id.clone();
        let tooltip_commit = rev.short_commit_id.clone();
        let author = rev.author_name.clone();
        let has_conflict = rev.has_conflict;
        let is_immutable = rev.is_immutable;
        let is_empty = rev.is_empty;

        h_flex()
            .id(id)
            .h(ROW_HEIGHT)
            .w_full()
            .px_0()
            .gap_2()
            .items_center()
            .cursor_pointer()
            .when(is_drag_target, |el| {
                // Give drag targets a crisp accent background so it's obvious
                // where the drop will land.
                el.bg(cx.theme().colors().drop_target_background)
            })
            .when(is_selected && !is_drag_target, |el| {
                el.bg(if is_focused {
                    cx.theme().colors().element_selected
                } else {
                    cx.theme().colors().element_hover
                })
            })
            .when(!is_selected && !is_drag_target, |el| {
                el.hover(|el| el.bg(cx.theme().colors().element_hover.opacity(0.5)))
            })
            .child(graph_cell)
            // ─── Workspace markers ───────────────────────────────────
            // Rendered as `@name` chips. With a single workspace `default` this
            // is redundant with the filled-circle node, so we skip that case.
            .children(
                workspaces
                    .into_iter()
                    .filter(|w| w.as_ref() != "default")
                    .map(move |ws| {
                        h_flex()
                            .gap_0p5()
                            .items_center()
                            .px_1()
                            .rounded_sm()
                            .bg(cx.theme().colors().element_background)
                            .border_1()
                            .border_color(cx.theme().colors().border_variant)
                            .child(
                                Label::new(SharedString::from(format!("@{ws}")))
                                    .size(LabelSize::XSmall)
                                    .color(Color::Accent),
                            )
                    }),
            )
            // ─── Bookmarks (locals draggable, remotes inert) ──────────
            .children(bookmarks.into_iter().enumerate().map(move |(bm_ix, bm)| {
                let label = bm.display();
                let is_local = bm.is_local();
                // Local: blue accent, grab cursor, draggable.
                // Remote (@origin): muted, default cursor, no drag.
                let (bg, border, fg) = if is_local {
                    (
                        cx.theme().status().info_background,
                        cx.theme().status().info_border,
                        Color::Info,
                    )
                } else {
                    (
                        cx.theme().colors().element_background,
                        cx.theme().colors().border_variant,
                        Color::Muted,
                    )
                };

                let chip = h_flex()
                    .id(("jj-bookmark", ix as u64 * 1000 + bm_ix as u64))
                    .gap_0p5()
                    .items_center()
                    .px_1()
                    .rounded_sm()
                    .bg(bg)
                    .border_1()
                    .border_color(border)
                    .child(
                        Icon::new(IconName::GitBranch)
                            .size(IconSize::XSmall)
                            .color(fg),
                    )
                    .child(Label::new(label.clone()).size(LabelSize::XSmall).color(fg));

                if is_local {
                    let bm_payload = DraggedJjBookmark {
                        name: bm.name,
                        from_change: change_id_for_bm.clone(),
                    };
                    chip.cursor_grab()
                        .on_drag(bm_payload, move |payload, click_offset, _, cx| {
                            cx.new(|_| DraggedJjBookmarkView {
                                name: payload.name.clone(),
                                click_offset,
                            })
                        })
                        .tooltip(Tooltip::text(format!("Drag to move `{label}`")))
                } else {
                    chip.tooltip(Tooltip::text(if bm.tracked {
                        format!("{label} (tracked remote, diverged)")
                    } else {
                        format!("{label} (untracked remote)")
                    }))
                }
            }))
            // ─── Description (main content, truncated) ─────────────────
            .child(
                div()
                    .flex_1()
                    .overflow_hidden()
                    .child(
                        Label::new(description)
                            .size(LabelSize::Small)
                            .color(description_color)
                            .single_line(),
                    ),
            )
            // ─── Flags & metadata ──────────────────────────────────────
            .when(has_conflict, |el| {
                el.child(
                    Icon::new(IconName::Warning)
                        .size(IconSize::XSmall)
                        .color(Color::Error),
                )
            })
            .when(is_empty && !is_immutable, |el| {
                el.child(
                    Label::new("(empty)")
                        .size(LabelSize::XSmall)
                        .color(Color::Muted),
                )
            })
            .child(
                Label::new(display_id)
                    .size(LabelSize::XSmall)
                    .color(Color::Muted),
            )
            .child(div().w_1())
            // Suppress the hover tooltip while the context menu is open — otherwise
            // it renders on top of the menu and obscures the top entries.
            .when(self.context_menu.is_none(), |el| {
                // change_id is in the visible column now; tooltip supplies the
                // commit SHA (volatile across rewrites) and author.
                el.tooltip(Tooltip::text(if author.is_empty() {
                    format!("commit {tooltip_commit}")
                } else {
                    format!("commit {tooltip_commit} · {author}")
                }))
            })
            // ─── Drag source: pick up a revision to rebase ─────────────
            .when(!is_immutable, |el| {
                let drag_payload = drag_payload.clone();
                el.on_drag(drag_payload, move |payload, click_offset, _, cx| {
                    cx.new(|_| DraggedJjRevisionView {
                        change_id: payload.change_id.clone(),
                        description: payload.description.clone(),
                        click_offset,
                    })
                })
            })
            // ─── Drop zone: receive a dragged revision (rebase) ────────
            .drag_over::<DraggedJjRevision>({
                let target = target_change.clone();
                move |el, dragged: &DraggedJjRevision, _, cx| {
                    // Don't highlight when dragging over ourselves.
                    if dragged.change_id == target {
                        el
                    } else {
                        el.bg(cx.theme().colors().drop_target_background)
                    }
                }
            })
            .on_drop(cx.listener(
                move |this, dragged: &DraggedJjRevision, _, cx| {
                    this.drag_target_ix = None;
                    let source = dragged.change_id.clone();
                    let dest = target_change_for_rev.clone();
                    // `-s`: subtree moves with the commit. `-r` (extract just
                    // this one) is the oddball case, reachable from the menu.
                    this.rebase_branch(source, dest, cx);
                },
            ))
            // ─── Drop zone: receive a dragged bookmark (move) ──────────
            .drag_over::<DraggedJjBookmark>({
                let target = target_change;
                move |el, dragged: &DraggedJjBookmark, _, cx| {
                    if dragged.from_change == target {
                        el
                    } else {
                        el.bg(cx.theme().colors().drop_target_background)
                    }
                }
            })
            .on_drop(cx.listener(
                move |this, dragged: &DraggedJjBookmark, _, cx| {
                    this.drag_target_ix = None;
                    let name = dragged.name.clone();
                    let to = target_change_for_bm.clone();
                    this.move_bookmark(name, to, cx);
                },
            ))
            // ─── Mouse handling ────────────────────────────────────────
            .on_click(cx.listener(move |this, event: &ClickEvent, _, cx| {
                if event.is_right_click() {
                    return;
                }
                this.selected_ix = Some(ix);
                if event.click_count() == 2 {
                    // Double-click → jj edit (like GG and jjui).
                    this.edit(target_change_for_click.clone(), cx);
                }
                cx.notify();
            }))
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                    this.deploy_context_menu(ix, event.position, window, cx);
                }),
            )
    }

    /// Paint one row-height slice of the graph: passthrough lines, incoming
    /// merge curves, the commit node, and outgoing edges.
    ///
    /// Drawing each row independently (rather than one tall canvas) means
    /// `uniform_list` virtualization, per-row hit testing, and drag/drop all
    /// work without any manual scroll bookkeeping.
    fn render_graph_cell(
        &self,
        entry: &JjLogEntry,
        graph_width: Pixels,
        cx: &Context<Self>,
    ) -> impl IntoElement + use<> {
        let lane = entry.lane;
        let passthrough = entry.passthrough_lanes.clone();
        let incoming = entry.incoming_lanes.clone();
        let edges: Vec<JjGraphEdge> = entry.edges.clone();
        let is_wc = entry.revision.is_working_copy;
        let is_immutable = entry.revision.is_immutable;

        // Color scheme: immutable commits use a muted neutral; mutable commits
        // cycle through the theme's accent colors by lane, so each branch gets
        // a consistent color down its length.
        let accents = cx.theme().accents().clone();
        let muted = cx.theme().colors().text_muted;
        let node_color = if is_immutable {
            muted
        } else {
            accents.color_for_index(lane as u32)
        };

        canvas(
            |_, _, _| {},
            move |bounds: Bounds<Pixels>, _: (), window: &mut Window, _: &mut App| {
                let top = bounds.origin.y;
                let bottom = bounds.origin.y + bounds.size.height;
                let mid_y = top + bounds.size.height / 2.0;

                let lane_x = |l: usize| -> Pixels {
                    bounds.origin.x + GRAPH_LEFT_PAD + LANE_WIDTH * l as f32
                };

                let line_color_for = |l: usize| -> Hsla { accents.color_for_index(l as u32) };

                // ─── Passthrough lanes: straight verticals top→bottom ──
                for l in &passthrough {
                    let x = lane_x(*l);
                    let mut b = PathBuilder::stroke(LINE_WIDTH);
                    b.move_to(point(x, top));
                    b.line_to(point(x, bottom));
                    if let Ok(p) = b.build() {
                        window.paint_path(p, line_color_for(*l));
                    }
                }

                let node_x = lane_x(lane);

                // ─── Incoming edges (from above, terminating here) ──────
                // Same-lane incoming → short vertical to the node.
                // Off-lane incoming → curve from (that_lane, top) to node.
                for l in &incoming {
                    let from_x = lane_x(*l);
                    let mut b = PathBuilder::stroke(LINE_WIDTH);
                    b.move_to(point(from_x, top));
                    if *l == lane {
                        b.line_to(point(node_x, mid_y - NODE_RADIUS));
                    } else {
                        // Quadratic curve via a control point that pulls the
                        // line horizontally first then down — matches the
                        // "elbow" look of `jj log` ASCII graphs.
                        let ctrl = point(from_x, mid_y);
                        b.curve_to(point(node_x, mid_y), ctrl);
                    }
                    if let Ok(p) = b.build() {
                        window.paint_path(p, line_color_for(*l));
                    }
                }

                // ─── Outgoing edges (toward parents below) ──────────────
                for edge in &edges {
                    let to_x = lane_x(edge.to_lane);
                    let mut b = PathBuilder::stroke(LINE_WIDTH);
                    if edge.to_lane == lane {
                        b.move_to(point(node_x, mid_y + NODE_RADIUS));
                        if edge.indirect {
                            // Parent is outside the revset — draw a stubby tail.
                            b.line_to(point(node_x, mid_y + NODE_RADIUS * 2.0));
                        } else {
                            b.line_to(point(node_x, bottom));
                        }
                    } else {
                        // Merge/branch: curve from node out to the target lane
                        // and down. Control point sits at the target x, node y
                        // so the curve flares outward immediately.
                        b.move_to(point(node_x, mid_y));
                        let ctrl = point(to_x, mid_y);
                        let end_y = if edge.indirect {
                            mid_y + NODE_RADIUS * 2.0
                        } else {
                            bottom
                        };
                        b.curve_to(point(to_x, end_y), ctrl);
                    }
                    if let Ok(p) = b.build() {
                        let color = if edge.indirect {
                            muted
                        } else {
                            line_color_for(edge.to_lane)
                        };
                        window.paint_path(p, color);
                    }
                }

                // ─── The commit node itself ─────────────────────────────
                // Working copy (@): filled circle, slightly larger.
                // Regular mutable: filled circle.
                // Immutable (◆): small filled diamond, muted.
                if is_immutable {
                    draw_diamond(node_x, mid_y, NODE_RADIUS * 0.9, node_color, window);
                } else if is_wc {
                    draw_circle(node_x, mid_y, NODE_RADIUS + px(1.0), node_color, window);
                    // Inner ring in bg color to give it a "target" look
                    // without needing a stroke-only path.
                } else {
                    draw_circle(node_x, mid_y, NODE_RADIUS, node_color, window);
                }
            },
        )
        .w(graph_width)
        .h(ROW_HEIGHT)
        .flex_none()
    }

    // `use<>` prevents the opaque return from capturing `cx`'s lifetime — nothing
    // in the returned element actually borrows from it (closures hold WeakEntity).
    // Without this, the borrow checker treats the returned header as holding a
    // live &mut cx and refuses further uses in Render::render.
    fn render_header(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let repo_name = self
            .repository
            .as_ref()
            .and_then(|r| {
                r.workspace_root()
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "Jujutsu".into());

        let busy = self.pending_mutation.is_some();

        h_flex()
            .w_full()
            .h(px(32.0))
            .px_2()
            .gap_1()
            .items_center()
            .border_b_1()
            .border_color(cx.theme().colors().border)
            .child(
                Icon::new(IconName::GitBranchAlt)
                    .size(IconSize::Small)
                    .color(Color::Muted),
            )
            .child(Label::new(repo_name).size(LabelSize::Small))
            .child(div().flex_1())
            .when(busy, |el| {
                el.child(
                    Label::new("…")
                        .size(LabelSize::XSmall)
                        .color(Color::Muted),
                )
            })
            .child(
                IconButton::new("jj-undo", IconName::Undo)
                    .icon_size(IconSize::Small)
                    .tooltip(Tooltip::text("Undo last jj operation"))
                    .on_click(cx.listener(|this, _, _, cx| this.undo(cx))),
            )
            .child(
                IconButton::new("jj-refresh", IconName::RotateCw)
                    .icon_size(IconSize::Small)
                    .tooltip(Tooltip::text("Refresh"))
                    .on_click(cx.listener(|this, _, _, cx| this.refresh(cx))),
            )
    }

    fn render_empty_state(&self, message: impl Into<SharedString>) -> impl IntoElement {
        v_flex()
            .size_full()
            .items_center()
            .justify_center()
            .gap_2()
            .child(
                Icon::new(IconName::GitBranchAlt)
                    .size(IconSize::XLarge)
                    .color(Color::Muted),
            )
            .child(
                Label::new(message)
                    .size(LabelSize::Small)
                    .color(Color::Muted),
            )
    }
}

// ─── Graph drawing primitives ───────────────────────────────────────────────

fn draw_circle(cx_px: Pixels, cy: Pixels, r: Pixels, color: Hsla, window: &mut Window) {
    let mut b = PathBuilder::fill();
    b.move_to(point(cx_px + r, cy));
    b.arc_to(point(r, r), px(0.), false, true, point(cx_px - r, cy));
    b.arc_to(point(r, r), px(0.), false, true, point(cx_px + r, cy));
    b.close();
    if let Ok(p) = b.build() {
        window.paint_path(p, color);
    }
}

fn draw_diamond(cx_px: Pixels, cy: Pixels, r: Pixels, color: Hsla, window: &mut Window) {
    let mut b = PathBuilder::fill();
    b.move_to(point(cx_px, cy - r));
    b.line_to(point(cx_px + r, cy));
    b.line_to(point(cx_px, cy + r));
    b.line_to(point(cx_px - r, cy));
    b.close();
    if let Ok(p) = b.build() {
        window.paint_path(p, color);
    }
}

// ─── Trait impls ────────────────────────────────────────────────────────────

impl Focusable for JjPanel {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<PanelEvent> for JjPanel {}

impl Panel for JjPanel {
    fn persistent_name() -> &'static str {
        "JjPanel"
    }

    fn panel_key() -> &'static str {
        JJ_PANEL_KEY
    }

    fn position(&self, _: &Window, _cx: &App) -> DockPosition {
        DockPosition::Left
    }

    fn position_is_valid(&self, position: DockPosition) -> bool {
        matches!(position, DockPosition::Left | DockPosition::Right)
    }

    fn set_position(&mut self, _position: DockPosition, _: &mut Window, _cx: &mut Context<Self>) {
        // Settings persistence is a follow-up — requires extending the
        // settings schema. For now position is fixed per session.
    }

    fn size(&self, _: &Window, _cx: &App) -> Pixels {
        self.width.unwrap_or(DEFAULT_PANEL_WIDTH)
    }

    fn set_size(&mut self, size: Option<Pixels>, _: &mut Window, cx: &mut Context<Self>) {
        self.width = size;
        cx.notify();
    }

    fn icon(&self, _: &Window, _cx: &App) -> Option<IconName> {
        // Only show the dock button if we actually found a .jj/ directory.
        // This keeps the panel invisible in pure-git repos and avoids
        // cluttering the dock.
        self.repository.as_ref().map(|_| IconName::GitBranchAlt)
    }

    fn icon_tooltip(&self, _: &Window, _: &App) -> Option<&'static str> {
        Some("Jujutsu Panel")
    }

    fn toggle_action(&self) -> Box<dyn Action> {
        Box::new(ToggleFocus)
    }

    fn activation_priority(&self) -> u32 {
        // Just after GitPanel (which is 2).
        3
    }

    fn enabled(&self, _cx: &App) -> bool {
        self.repository.is_some()
    }
}

impl Render for JjPanel {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let panel_bg = cx.theme().colors().panel_background;
        let header = self.render_header(cx);

        let body: gpui::AnyElement = match &self.state {
            PanelState::NotAJjRepo => self
                .render_empty_state("No .jj/ directory found in project")
                .into_any_element(),
            PanelState::Loading => self
                .render_empty_state("Loading jj log…")
                .into_any_element(),
            PanelState::Error(msg) => v_flex()
                .size_full()
                .p_4()
                .gap_2()
                .child(
                    Label::new("Failed to load jj log")
                        .size(LabelSize::Small)
                        .color(Color::Error),
                )
                .child(
                    Label::new(msg.clone())
                        .size(LabelSize::XSmall)
                        .color(Color::Muted),
                )
                .into_any_element(),
            PanelState::Loaded {
                entries, max_lanes, ..
            } => {
                let entries = entries.clone();
                let count = entries.len();
                let graph_width =
                    GRAPH_LEFT_PAD + LANE_WIDTH * (*max_lanes).max(1) as f32 + GRAPH_RIGHT_PAD;

                uniform_list(
                    "jj-log-entries",
                    count,
                    cx.processor(move |this, range: Range<usize>, window, cx| {
                        let mut rows = Vec::with_capacity(range.end - range.start);
                        for ix in range {
                            if let Some(entry) = entries.get(ix) {
                                rows.push(
                                    this.render_row(ix, entry, graph_width, window, cx)
                                        .into_any_element(),
                                );
                            }
                        }
                        rows
                    }),
                )
                .track_scroll(&self.scroll_handle)
                .size_full()
                .into_any_element()
            }
        };

        v_flex()
            .id("jj-panel")
            .key_context("JjPanel")
            .track_focus(&self.focus_handle)
            .size_full()
            .bg(panel_bg)
            .on_action(cx.listener(|this, _: &Refresh, _, cx| this.refresh(cx)))
            .child(header)
            .child(body)
            .children(self.context_menu.as_ref().map(|(menu, position, _)| {
                deferred(
                    anchored()
                        .position(*position)
                        .anchor(Corner::TopLeft)
                        .child(menu.clone()),
                )
                .with_priority(1)
            }))
    }
}
