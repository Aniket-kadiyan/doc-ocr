/** A user-defined symbol from the shared backend library. */
export interface CustomSymbol {
  id: string;
  name: string;
  /** Balloon label for every match; empty means the name is used. */
  label: string;
  category: string;
  /** Match strictness: the correlation a match must reach (0.5–0.99). */
  threshold: number;
  /** Also look for the symbol turned 90°, 180° and 270°. */
  match_rotations: boolean;
  /** Search every size rather than sizes near the uploaded crop's. */
  any_size: boolean;
  enabled: boolean;
  width: number;
  height: number;
  created_at: string;
  updated_at: string;
  created_by?: string | null;
  open_sides?: string[];
  /** The cleaned template exactly as it is matched, dark on white. */
  image_data_url?: string;
}

export interface CustomSymbolLibrary {
  symbols: CustomSymbol[];
  categories: string[];
  defaults: {
    category: string;
    threshold: number;
    min_threshold: number;
    max_threshold: number;
  };
}

export interface NewCustomSymbol {
  file: File;
  name: string;
  label: string;
  category: string;
  threshold: number;
  matchRotations: boolean;
  anySize: boolean;
}

export type CustomSymbolUpdate = Partial<
  Pick<
    CustomSymbol,
    | "name"
    | "label"
    | "category"
    | "threshold"
    | "match_rotations"
    | "any_size"
    | "enabled"
  >
>;
