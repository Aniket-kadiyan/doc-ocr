import Dexie, { type Table } from "dexie";
import type { Annotation } from "@/types/annotation";
import type { ScanCandidate } from "@/types/scanCandidate";
import {
  normalizeDocumentMetadata,
  type DocumentMetadata,
} from "@/types/documentMetadata";

export interface ProjectRecord {
  id: string;
  name: string;
  fileName: string;
  fileType: "pdf" | "image";
  /** Original drawing bytes, stored so the PDF/image reopens with its annotations. */
  fileBlob?: Blob;
  mimeType?: string;
  /** Optional for compatibility with projects saved before metadata support. */
  metadata?: DocumentMetadata;
  updatedAt: number;
}

export interface StoredAnnotation extends Annotation {
  projectId: string;
}

export interface StoredScanCandidate extends ScanCandidate {
  projectId: string;
}

export class DrawingDB extends Dexie {
  annotations!: Table<StoredAnnotation, string>;
  scanCandidates!: Table<StoredScanCandidate, string>;
  projects!: Table<ProjectRecord, string>;

  constructor() {
    super("DrawingAnnotationDB");
    this.version(1).stores({
      annotations: "id, projectId, page, number, createdAt",
      projects: "id, name, updatedAt",
    });
    this.version(2).stores({
      annotations: "id, projectId, page, number, createdAt",
      scanCandidates: "id, projectId, page, state, order, updatedAt",
      projects: "id, name, updatedAt",
    });
  }
}

export const db = new DrawingDB();

export async function saveAnnotations(
  projectId: string,
  annotations: Annotation[]
): Promise<void> {
  await db.transaction("rw", db.annotations, async () => {
    const existing = await db.annotations
      .where("projectId")
      .equals(projectId)
      .toArray();
    await db.annotations.bulkDelete(existing.map((a) => a.id));
    await db.annotations.bulkPut(
      annotations.map((ann) => ({ ...ann, projectId }))
    );
  });
}

export async function loadAnnotations(
  projectId: string
): Promise<Annotation[]> {
  const stored = await db.annotations
    .where("projectId")
    .equals(projectId)
    .toArray();
  return stored.map(({ projectId, ...ann }) => {
    void projectId;
    return ann;
  });
}

export async function saveScanCandidates(
  projectId: string,
  candidates: ScanCandidate[]
): Promise<void> {
  await db.transaction("rw", db.scanCandidates, async () => {
    const existing = await db.scanCandidates
      .where("projectId")
      .equals(projectId)
      .toArray();
    await db.scanCandidates.bulkDelete(
      existing.map((candidate) => candidate.id)
    );
    await db.scanCandidates.bulkPut(
      candidates.map((candidate) => ({ ...candidate, projectId }))
    );
  });
}

export async function loadScanCandidates(
  projectId: string
): Promise<ScanCandidate[]> {
  const stored = await db.scanCandidates
    .where("projectId")
    .equals(projectId)
    .toArray();
  return stored
    .map(({ projectId: _projectId, ...candidate }) => {
      void _projectId;
      return candidate;
    })
    .sort((left, right) => left.order - right.order);
}

export async function saveProject(project: ProjectRecord): Promise<void> {
  await db.projects.put(project);
}

export async function saveProjectMetadata(
  projectId: string,
  metadata: DocumentMetadata
): Promise<void> {
  await db.projects.update(projectId, {
    metadata: normalizeDocumentMetadata(metadata),
    updatedAt: Date.now(),
  });
}

export async function getProject(
  projectId: string
): Promise<ProjectRecord | undefined> {
  return db.projects.get(projectId);
}

/** Remove a project with its accepted annotations and candidate audit records. */
export async function deleteProject(projectId: string): Promise<void> {
  await db.transaction(
    "rw",
    db.projects,
    db.annotations,
    db.scanCandidates,
    async () => {
      await db.projects.delete(projectId);
      const owned = await db.annotations
        .where("projectId")
        .equals(projectId)
        .toArray();
      await db.annotations.bulkDelete(owned.map((a) => a.id));
      const ownedCandidates = await db.scanCandidates
        .where("projectId")
        .equals(projectId)
        .toArray();
      await db.scanCandidates.bulkDelete(
        ownedCandidates.map((candidate) => candidate.id)
      );
    }
  );
}

export async function listProjects(): Promise<ProjectRecord[]> {
  return db.projects.orderBy("updatedAt").reverse().toArray();
}

/** Most recently saved project — used to reopen the last drawing on app load. */
export async function getMostRecentProject(): Promise<
  ProjectRecord | undefined
> {
  return db.projects.orderBy("updatedAt").reverse().first();
}
