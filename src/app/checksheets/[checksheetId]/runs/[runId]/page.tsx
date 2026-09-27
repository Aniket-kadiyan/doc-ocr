import { ChecksheetRun } from "@/components/checksheets/ChecksheetRun";

interface ChecksheetRunPageProps {
  params: Promise<{
    checksheetId: string;
    runId: string;
  }>;
}

export default async function ChecksheetRunPage({
  params,
}: ChecksheetRunPageProps) {
  const { checksheetId, runId } = await params;
  return <ChecksheetRun checksheetId={checksheetId} runId={runId} />;
}
