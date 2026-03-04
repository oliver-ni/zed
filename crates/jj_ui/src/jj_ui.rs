//! Native Jujutsu (jj) integration for Zed.
//!
//! Provides a dockable panel showing the `jj log` graph with drag-to-rebase
//! and drag-to-move-bookmark interactions.

mod jj_panel;

pub use jj_panel::JjPanel;

use gpui::{App, actions};
use workspace::Workspace;

actions!(
    jj_panel,
    [
        /// Toggles focus on the Jujutsu panel.
        ToggleFocus,
        /// Refresh the Jujutsu log.
        Refresh,
    ]
);

pub fn init(cx: &mut App) {
    cx.observe_new(|workspace: &mut Workspace, _, _cx| {
        workspace.register_action(|workspace, _: &ToggleFocus, window, cx| {
            workspace.toggle_panel_focus::<JjPanel>(window, cx);
        });
    })
    .detach();
}
