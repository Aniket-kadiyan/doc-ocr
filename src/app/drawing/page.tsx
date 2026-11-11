import type { Metadata } from "next";
import { DrawingViewerLoader } from "@/components/DrawingViewerLoader";

export const metadata: Metadata = {
    title: "Drawing Workspace | QTraX",
};

export default function DrawingWorkspacePage() {
    return <DrawingViewerLoader />;
}