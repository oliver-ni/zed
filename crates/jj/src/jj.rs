//! Jujutsu (jj) VCS backend.
//!
//! Shells out to the `jj` CLI using structured template output.
//! Computes a lane-based graph layout suitable for rendering in a commit graph view.

use anyhow::{Context as _, Result, bail};
use collections::HashMap;
use gpui::SharedString;
use std::path::{Path, PathBuf};
use util::command::new_command;

/// ASCII Unit Separator — delimits fields within a record.
const FS: char = '\x1f';
/// ASCII Record Separator — delimits records.
const RS: char = '\x1e';

/// The `jj log` template. Each record is terminated by RS+newline; fields are
/// separated by FS. Field order matches [`JjRevision`] parsing.
///
/// We use full-length commit IDs for parent references so that edges resolve
/// unambiguously even in very large repos, but display short prefixes.
const LOG_TEMPLATE: &str = r#"change_id.shortest(12) ++ "\x1f" ++ commit_id ++ "\x1f" ++ commit_id.shortest(12) ++ "\x1f" ++ description.first_line() ++ "\x1f" ++ author.name() ++ "\x1f" ++ author.email() ++ "\x1f" ++ author.timestamp().format("%s") ++ "\x1f" ++ if(current_working_copy, "1", "0") ++ "\x1f" ++ if(immutable, "1", "0") ++ "\x1f" ++ if(conflict, "1", "0") ++ "\x1f" ++ if(empty, "1", "0") ++ "\x1f" ++ bookmarks.join(",") ++ "\x1f" ++ parents.map(|p| p.commit_id()).join(",") ++ "\x1e\n""#;

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
    /// Local bookmarks pointing at this revision.
    pub bookmarks: Vec<SharedString>,
    /// Full commit IDs of parents.
    pub parent_ids: Vec<SharedString>,
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
/// edge travels to for the *next* row. If the parent is not in the query
/// result set, the edge is marked `indirect` and the line should be drawn
/// dashed / terminated early.
#[derive(Debug, Clone)]
pub struct JjGraphEdge {
    pub from_lane: usize,
    pub to_lane: usize,
    /// Full commit ID of the parent this edge points to.
    pub to_commit: SharedString,
    /// True if the parent is outside the current revset (edge goes "off the
    /// bottom" of what we're showing).
    pub indirect: bool,
}

/// A handle for running `jj` commands against a specific workspace.
///
/// This is deliberately stateless — each operation re-reads from disk. The
/// `jj` CLI is fast enough that this is fine for a panel that refreshes on
/// file events, and it means we never display stale operation-log state.
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

    /// Run `jj log` with the given revset and return laid-out graph rows.
    ///
    /// The default revset `::` includes all reachable commits. For large repos
    /// you probably want something narrower like `::@ | trunk()` or the user's
    /// configured default revset.
    pub async fn log(&self, revset: &str) -> Result<Vec<JjLogEntry>> {
        let output = new_command("jj")
            .current_dir(&self.workspace_root)
            .args(["--no-pager", "--color=never", "log", "--no-graph", "-r"])
            .arg(revset)
            .arg("-T")
            .arg(LOG_TEMPLATE)
            .output()
            .await
            .context("running jj log")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("jj log failed: {}", stderr.trim());
        }

        let stdout = String::from_utf8(output.stdout)
            .context("jj log output was not valid UTF-8")?;

        let revisions = parse_log_output(&stdout)?;
        Ok(layout_graph(revisions))
    }

    /// Get the user's configured default revset for `jj log`. Falls back to a
    /// reasonable default if the config lookup fails.
    pub async fn default_revset(&self) -> String {
        let output = new_command("jj")
            .current_dir(&self.workspace_root)
            .args(["--no-pager", "config", "get", "revsets.log"])
            .output()
            .await;

        match output {
            Ok(out) if out.status.success() => {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if s.is_empty() {
                    "present(@) | ancestors(immutable_heads().., 2) | trunk()".to_string()
                } else {
                    s
                }
            }
            _ => "present(@) | ancestors(immutable_heads().., 2) | trunk()".to_string(),
        }
    }

    /// `jj rebase -r <source> -d <destination>`
    ///
    /// Moves a single revision onto a new parent. Descendants are
    /// automatically rebased to follow.
    pub async fn rebase_revision(&self, source_change: &str, dest_change: &str) -> Result<()> {
        self.run_mutation(&["rebase", "-r", source_change, "-d", dest_change])
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

fn parse_log_output(stdout: &str) -> Result<Vec<JjRevision>> {
    let mut revisions = Vec::new();
    for record in stdout.split(RS) {
        let record = record.trim_matches('\n');
        if record.is_empty() {
            continue;
        }
        let fields: Vec<&str> = record.split(FS).collect();
        if fields.len() != 13 {
            // Tolerate trailing garbage / partial writes rather than failing
            // the whole panel.
            log::warn!("skipping malformed jj log record ({} fields): {:?}", fields.len(), record);
            continue;
        }
        let bookmarks = if fields[11].is_empty() {
            Vec::new()
        } else {
            fields[11]
                .split(',')
                .map(|s| SharedString::from(s.to_string()))
                .collect()
        };
        let parent_ids = if fields[12].is_empty() {
            Vec::new()
        } else {
            fields[12]
                .split(',')
                .map(|s| SharedString::from(s.to_string()))
                .collect()
        };
        revisions.push(JjRevision {
            change_id: fields[0].to_string().into(),
            commit_id: fields[1].to_string().into(),
            short_commit_id: fields[2].to_string().into(),
            description: fields[3].to_string().into(),
            author_name: fields[4].to_string().into(),
            author_email: fields[5].to_string().into(),
            timestamp: fields[6].parse().unwrap_or(0),
            is_working_copy: fields[7] == "1",
            is_immutable: fields[8] == "1",
            has_conflict: fields[9] == "1",
            is_empty: fields[10] == "1",
            bookmarks,
            parent_ids,
        });
    }
    Ok(revisions)
}

/// Assign each revision to a vertical lane and compute the outgoing edges.
///
/// This is a classic "stem"-based layout: we maintain a list of open lanes,
/// each waiting for a particular parent commit. When a commit arrives, it
/// claims the first lane that was waiting for it (terminating any other lanes
/// waiting for the same commit as merges). Its parents then reserve lanes for
/// the rows below.
///
/// `jj log` already outputs in topological order (children before parents),
/// which is exactly what this algorithm needs.
fn layout_graph(revisions: Vec<JjRevision>) -> Vec<JjLogEntry> {
    // Set of commit IDs present in this log — used to mark edges that point
    // outside the revset as indirect.
    let present: collections::HashSet<_> = revisions
        .iter()
        .map(|r| r.commit_id.clone())
        .collect();

    // Each slot holds the commit ID it's waiting for, or None if free.
    let mut lanes: Vec<Option<SharedString>> = Vec::new();
    let mut entries = Vec::with_capacity(revisions.len());

    for rev in revisions {
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

        // Reserve lanes for parents. First parent stays in our lane; additional
        // parents (merges) spawn new lanes.
        let mut edges = Vec::with_capacity(rev.parent_ids.len());
        for (parent_idx, parent_id) in rev.parent_ids.iter().enumerate() {
            let indirect = !present.contains(parent_id);

            let to_lane = if parent_idx == 0 {
                // First parent continues in our lane.
                if !indirect {
                    if lanes.len() <= my_lane {
                        lanes.resize(my_lane + 1, None);
                    }
                    lanes[my_lane] = Some(parent_id.clone());
                }
                my_lane
            } else {
                // Merge parent: check if some other lane is already waiting for
                // this same parent (common ancestor). If so, join it instead of
                // spawning a duplicate.
                let existing = lanes
                    .iter()
                    .position(|w| w.as_ref() == Some(parent_id));
                if let Some(existing_lane) = existing {
                    existing_lane
                } else if indirect {
                    // Parent is outside the revset; don't reserve a real lane.
                    // Point the edge at a phantom lane one past the end for
                    // rendering purposes.
                    lanes.len()
                } else {
                    // Spawn a new lane.
                    match lanes.iter().position(Option::is_none) {
                        Some(i) => {
                            lanes[i] = Some(parent_id.clone());
                            i
                        }
                        None => {
                            lanes.push(Some(parent_id.clone()));
                            lanes.len() - 1
                        }
                    }
                }
            };

            edges.push(JjGraphEdge {
                from_lane: my_lane,
                to_lane,
                to_commit: parent_id.clone(),
                indirect,
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
        .flat_map(|e| {
            std::iter::once(e.lane).chain(e.edges.iter().map(|edge| edge.to_lane))
        })
        .max()
        .map(|m| m + 1)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single() {
        let input = "abc12345\x1ffull_commit_id_here\x1fabc123\x1fhello world\x1fAlice\x1falice@example.com\x1f1234567890\x1f1\x1f0\x1f0\x1f0\x1fmain,dev\x1fparent_full_id\x1e\n";
        let revs = parse_log_output(input).unwrap();
        assert_eq!(revs.len(), 1);
        let r = &revs[0];
        assert_eq!(r.change_id.as_ref(), "abc12345");
        assert_eq!(r.description.as_ref(), "hello world");
        assert!(r.is_working_copy);
        assert!(!r.is_immutable);
        assert_eq!(r.bookmarks.len(), 2);
        assert_eq!(r.parent_ids.len(), 1);
    }

    #[test]
    fn test_layout_linear() {
        // c → b → a (c is newest, a is oldest)
        let revs = vec![
            rev("c", vec!["b"]),
            rev("b", vec!["a"]),
            rev("a", vec![]),
        ];
        let entries = layout_graph(revs);
        assert_eq!(entries.len(), 3);
        // All should be in lane 0.
        for e in &entries {
            assert_eq!(e.lane, 0);
        }
    }

    #[test]
    fn test_layout_merge() {
        // Merge: d has two parents b and c, which both descend from a.
        //   d
        //  / \
        // b   c
        //  \ /
        //   a
        let revs = vec![
            rev("d", vec!["b", "c"]),
            rev("b", vec!["a"]),
            rev("c", vec!["a"]),
            rev("a", vec![]),
        ];
        let entries = layout_graph(revs);
        assert_eq!(entries[0].lane, 0); // d
        assert_eq!(entries[0].edges.len(), 2);
        // b should be in lane 0 (first parent of d)
        assert_eq!(entries[1].lane, 0);
        // c should be in lane 1 (second parent spawned new lane)
        assert_eq!(entries[2].lane, 1);
        // a should merge both back to lane 0
        assert_eq!(entries[3].lane, 0);
    }

    fn rev(id: &str, parents: Vec<&str>) -> JjRevision {
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
            parent_ids: parents.iter().map(|p| p.to_string().into()).collect(),
        }
    }
}
