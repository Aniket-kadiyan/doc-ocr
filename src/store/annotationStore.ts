import { create } from "zustand";
import type { Annotation, PendingSelection } from "@/types/annotation";

const isValue = (annotation: Annotation) => annotation.kind !== "label";

interface AnnotationState {
  annotations: Annotation[];
  pending: PendingSelection | null;
  currentPage: number;
  totalPages: number;
  scale: number;
  isSegmenting: boolean;
  /** The Draw Value tool is active and waiting for one box. */
  isDrawingValue: boolean;
  /** id of the value whose metadata editor is open, or null. */
  editingValueId: string | null;
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
  setIsDrawingValue: (drawing: boolean) => void;
  setEditingValueId: (id: string | null) => void;
  setIsProcessing: (processing: boolean) => void;
  setProjectName: (name: string) => void;
  setProjectId: (id: string) => void;
  /** Next visible balloon number. Deleted numbers remain as gaps. */
  getNextNumber: () => number;
}

export const useAnnotationStore = create<AnnotationState>((set, get) => ({
  annotations: [],
  pending: null,
  currentPage: 1,
  totalPages: 1,
  scale: 1,
  isSegmenting: false,
  isDrawingValue: false,
  editingValueId: null,
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
      const existingNumbers = state.annotations
        .filter(isValue)
        .map((annotation) => annotation.number);
      let nextNumber = existingNumbers.length
        ? Math.max(...existingNumbers) + 1
        : 1;
      const numbered = incoming.filter(isValue).map((annotation) => ({
        ...annotation,
        kind: "dimension" as const,
        number: nextNumber++,
      }));
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
      editingValueId:
        state.editingValueId === id ? null : state.editingValueId,
    })),

  setPending: (pending) => set({ pending }),

  setCurrentPage: (page) => set({ currentPage: page }),

  setTotalPages: (total) => set({ totalPages: total }),

  setScale: (scale) => set({ scale }),

  setIsSegmenting: (isSegmenting) => set({ isSegmenting }),

  setIsDrawingValue: (isDrawingValue) => set({ isDrawingValue }),

  setEditingValueId: (editingValueId) => set({ editingValueId }),

  setIsProcessing: (isProcessing) => set({ isProcessing }),

  setProjectName: (projectName) => set({ projectName }),

  setProjectId: (projectId) => set({ projectId }),

  getNextNumber: () => {
    const nums = get()
      .annotations.filter(isValue)
      .map((a) => a.number);
    return nums.length ? Math.max(...nums) + 1 : 1;
  },
}));
