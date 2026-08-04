/**
 * Feature flags for development-only tools (Auto-Segment, OCR Debug dump).
 *
 * Both are dev conveniences. In a production build they are HIDDEN and INERT —
 * the buttons don't render and no debug dump is ever produced.
 */

/**
 * Auto-Segment is disabled for now, everywhere (dev and prod). Flip this back
 * to `process.env.NODE_ENV !== "production"` to restore the dev-only behavior.
 */
export const AUTO_SEGMENT_ENABLED = false;

/**
 * OCR Debug dump is disabled for now, everywhere (dev and prod). The button
 * never renders and the dump function is hard-gated off. Flip this back to
 * `process.env.NODE_ENV !== "production"` to restore the dev-only behavior.
 */
export const DEBUG_DUMP_ENABLED = false;
