import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { accessDenied, ownerAuthorized, privateModeConfigured, privateModeEnabled } from "./app/lib/owner-auth";

export function proxy(request: NextRequest) {
  if (!privateModeEnabled()) return NextResponse.next();
  if (!privateModeConfigured()) {
    return NextResponse.json({ detail: { code: "PRIVATE_NOT_CONFIGURED", message: "Owner access is not configured" } }, { status: 503, headers: { "Cache-Control": "no-store" } });
  }
  if (!ownerAuthorized(request.headers.get("authorization"))) return accessDenied();
  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
