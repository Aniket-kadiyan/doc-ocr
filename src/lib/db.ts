import Dexie, { type Table } from "dexie";
import type { Annotation } from "@/types/annotation";

export interface ProjectRecord {
  id: string;
  name: string;
  fileName: string;
  fileType: "pdf" | "image";
  /** Original drawing bytes, stored so the PDF/image reopens with its annotations. */
  fileBlob?: Blob;
  mimeType?: string;
  updatedAt: number;
}

export interface StoredAnnotation extends Annotation {
  projectId: string;
}

export class DrawingDB extends Dexie {
  annotations!: Table<StoredAnnotation, string>;
  projects!: Table<ProjectRecord, string>;

  constructor() {
    super("DrawingAnnotationDB");
    this.version(1).stores({
      annotations: "id, projectId, page, number, createdAt",
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

export async function saveProject(project: ProjectRecord): Promise<void> {
  await db.projects.put(project);
}

export async function getProject(
  projectId: string
): Promise<ProjectRecord | undefined> {
  return db.projects.get(projectId);
}

/** Remove a project and all of its annotations (e.g. when closing a drawing). */
export async function deleteProject(projectId: string): Promise<void> {
  await db.transaction("rw", db.projects, db.annotations, async () => {
    await db.projects.delete(projectId);
    const owned = await db.annotations
      .where("projectId")
      .equals(projectId)
      .toArray();
    await db.annotations.bulkDelete(owned.map((a) => a.id));
  });
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
