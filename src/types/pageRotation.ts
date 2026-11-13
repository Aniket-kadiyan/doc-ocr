export const PAGE_ROTATIONS = [0, 90, 180, 270] as const;

export type PageRotation = (typeof PAGE_ROTATIONS)[number];

/** One clockwise quarter-turn value per 1-based document page. */
export type PageRotations = Record<number, PageRotation>;
