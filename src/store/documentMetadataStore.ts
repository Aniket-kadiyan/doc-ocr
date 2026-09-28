import { create } from "zustand";
import {
  normalizeDocumentMetadata,
  type DocumentMetadata,
  type DocumentMetadataField,
} from "@/types/documentMetadata";

interface DocumentMetadataState {
  metadata: DocumentMetadata;
  setMetadata: (metadata?: Partial<DocumentMetadata> | null) => void;
  updateField: (field: DocumentMetadataField, value: string) => void;
  resetMetadata: () => void;
}

export const useDocumentMetadataStore = create<DocumentMetadataState>(
  (set) => ({
    metadata: normalizeDocumentMetadata(),
    setMetadata: (metadata) =>
      set({ metadata: normalizeDocumentMetadata(metadata) }),
    updateField: (field, value) =>
      set((state) => ({
        metadata: { ...state.metadata, [field]: value },
      })),
    resetMetadata: () => set({ metadata: normalizeDocumentMetadata() }),
  })
);
