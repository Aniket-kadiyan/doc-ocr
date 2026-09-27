/**
 * Feature flags for development-only tools.
 *
 * Debug output is hidden and inert in production builds.
 */

/**
 * OCR Debug dump is disabled for now, everywhere (dev and prod). The button
 * never renders and the dump function is hard-gated off. Flip this back to
 * `process.env.NODE_ENV !== "production"` to restore the dev-only behavior.
 */
export const DEBUG_DUMP_ENABLED = false;

/**
 * Temporary scan visualization requested for panel/table-mask tuning.
 * Set this single code flag to false once layout validation is complete.
 */
export const SCAN_DEBUG_OVERLAY_ENABLED = true;
