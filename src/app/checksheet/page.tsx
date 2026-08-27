import { redirect } from "next/navigation";

export default function LegacyChecksheetRedirect() {
  redirect("/checksheets");
}
