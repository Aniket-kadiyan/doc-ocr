import { create } from "zustand";
import {
  moveValueAnnotationToNumber,
  renumberValueAnnotations,
} from "@/lib/annotationNumbers";
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
  /** id of the open IndexedDB project used to retrieve its source drawing. */
  projectId: string;

  setAnnotations: (annotations: Annotation[]) => void;
  addAnnotation: (annotation: Annotation) => void;
  addAnnotations: (annotations: Annotation[]) => void;
  updateAnnotation: (id: string, patch: Partial<Annotation>) => void;
  moveAnnotationToNumber: (id: string, targetNumber: number) => void;
  setAllBalloonVisibility: (visible: boolean) => void;
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
  /** Next visible balloon number in the contiguous 1…N sequence. */
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

  setAnnotations: (annotations) =>
    set({ annotations: renumberValueAnnotations(annotations) }),

  addAnnotation: (annotation) =>
    set((state) => ({
      annotations: renumberValueAnnotations([
        ...state.annotations,
        annotation,
      ]),
      pending: null,
    })),

  addAnnotations: (incoming) =>
    set((state) => {
      if (incoming.length === 0) return {};
      let nextNumber = state.annotations.filter(isValue).length + 1;
      const numbered = incoming.filter(isValue).map((annotation) => ({
        ...annotation,
        kind: "dimension" as const,
        number: nextNumber++,
      }));
      return {
        annotations: renumberValueAnnotations([
          ...state.annotations,
          ...numbered,
        ]),
        pending: null,
      };
    }),

  updateAnnotation: (id, patch) =>
    set((state) => ({
      annotations: state.annotations.map((a) =>
        a.id === id ? { ...a, ...patch } : a
      ),
    })),

  moveAnnotationToNumber: (id, targetNumber) =>
    set((state) => ({
      annotations: moveValueAnnotationToNumber(
        state.annotations,
        id,
        targetNumber
      ),
    })),

  setAllBalloonVisibility: (visible) =>
    set((state) => ({
      annotations: state.annotations.map((annotation) =>
        isValue(annotation)
          ? { ...annotation, hidden: !visible }
          : annotation
      ),
    })),

  removeAnnotation: (id) =>
    set((state) => ({
      annotations: renumberValueAnnotations(
        state.annotations.filter((a) => a.id !== id)
      ),
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
    return get().annotations.filter(isValue).length + 1;
  },
}));
