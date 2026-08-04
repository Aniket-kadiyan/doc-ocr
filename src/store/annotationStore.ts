import { create } from "zustand";
import type {
  Annotation,
  AnnotationKind,
  LabelInputMode,
  PendingSelection,
} from "@/types/annotation";

/** Treat a missing kind as "dimension" (older annotations predate the field). */
const kindOf = (a: Annotation): AnnotationKind => a.kind ?? "dimension";

interface AnnotationState {
  annotations: Annotation[];
  pending: PendingSelection | null;
  currentPage: number;
  totalPages: number;
  scale: number;
  isSegmenting: boolean;
  /** The "Add Label" tool is active — draw the box for a label. */
  isLabeling: boolean;
  /** When labeling, whether the label is typed or OCR'd from the drawn box. */
  labelInputMode: LabelInputMode;
  /** id of the label a value is being added to (draw the box to OCR), or null. */
  addValueLabelId: string | null;
  /** id of the label whose values are open in the edit modal, or null. */
  editingLabelId: string | null;
  isProcessing: boolean;
  projectName: string;
  /** id of the open project in IndexedDB — shared with the checksheet web view. */
  projectId: string;

  setAnnotations: (annotations: Annotation[]) => void;
  addAnnotation: (annotation: Annotation) => void;
  addAnnotations: (annotations: Annotation[]) => void;
  updateAnnotation: (id: string, patch: Partial<Annotation>) => void;
  removeAnnotation: (id: string) => void;
  setPending: (pending: PendingSelection | null) => void;
  setCurrentPage: (page: number) => void;
  setTotalPages: (total: number) => void;
  setScale: (scale: number) => void;
  setIsSegmenting: (segmenting: boolean) => void;
  setIsLabeling: (labeling: boolean) => void;
  setLabelInputMode: (mode: LabelInputMode) => void;
  setAddValueLabelId: (id: string | null) => void;
  setEditingLabelId: (id: string | null) => void;
  setIsProcessing: (processing: boolean) => void;
  setProjectName: (name: string) => void;
  setProjectId: (id: string) => void;
  /** Next sequence number within a kind, so dimensions and labels each count 1,2,3… */
  getNextNumber: (kind?: AnnotationKind) => number;
}

export const useAnnotationStore = create<AnnotationState>((set, get) => ({
  annotations: [],
  pending: null,
  currentPage: 1,
  totalPages: 1,
  scale: 1,
  isSegmenting: false,
  isLabeling: false,
  labelInputMode: "manual",
  addValueLabelId: null,
  editingLabelId: null,
  isProcessing: false,
  projectName: "Untitled Drawing",
  projectId: "",

  setAnnotations: (annotations) => set({ annotations }),

  addAnnotation: (annotation) =>
    set((state) => ({
      annotations: [...state.annotations, annotation],
      pending: null,
    })),

  addAnnotations: (incoming) =>
    set((state) => {
      if (incoming.length === 0) return {};
      // Number each batch within its own kind so dimensions and labels keep
      // independent 1,2,3… sequences and don't collide.
      const maxByKind = (kind: AnnotationKind) => {
        const nums = state.annotations
          .filter((a) => kindOf(a) === kind)
          .map((a) => a.number);
        return nums.length ? Math.max(...nums) : 0;
      };
      const counters: Record<AnnotationKind, number> = {
        dimension: maxByKind("dimension"),
        label: maxByKind("label"),
      };
      const numbered = incoming.map((a) => {
        const kind = kindOf(a);
        counters[kind] += 1;
        return { ...a, number: counters[kind] };
      });
      return { annotations: [...state.annotations, ...numbered], pending: null };
    }),

  updateAnnotation: (id, patch) =>
    set((state) => ({
      annotations: state.annotations.map((a) =>
        a.id === id ? { ...a, ...patch } : a
      ),
    })),

  removeAnnotation: (id) =>
    set((state) => ({
      annotations: state.annotations.filter((a) => a.id !== id),
    })),

  setPending: (pending) => set({ pending }),

  setCurrentPage: (page) => set({ currentPage: page }),

  setTotalPages: (total) => set({ totalPages: total }),

  setScale: (scale) => set({ scale }),

  setIsSegmenting: (isSegmenting) => set({ isSegmenting }),

  setIsLabeling: (isLabeling) => set({ isLabeling }),

  setLabelInputMode: (labelInputMode) => set({ labelInputMode }),

  setAddValueLabelId: (addValueLabelId) => set({ addValueLabelId }),

  setEditingLabelId: (editingLabelId) => set({ editingLabelId }),

  setIsProcessing: (isProcessing) => set({ isProcessing }),

  setProjectName: (projectName) => set({ projectName }),

  setProjectId: (projectId) => set({ projectId }),

  getNextNumber: (kind = "dimension") => {
    const nums = get()
      .annotations.filter((a) => kindOf(a) === kind)
      .map((a) => a.number);
    return nums.length ? Math.max(...nums) + 1 : 1;
  },
}));
