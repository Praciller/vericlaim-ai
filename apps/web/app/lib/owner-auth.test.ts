import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { ownerAuthorized, privateModeConfigured, privateModeEnabled } from "./owner-auth.ts";

const previous = {
  VERICLAIM_PRIVATE_MODE: process.env.VERICLAIM_PRIVATE_MODE,
  VERICLAIM_OWNER_USER: process.env.VERICLAIM_OWNER_USER,
  VERICLAIM_OWNER_PASSWORD: process.env.VERICLAIM_OWNER_PASSWORD,
  VERICLAIM_API_ACCESS_TOKEN: process.env.VERICLAIM_API_ACCESS_TOKEN,
};
afterEach(() => {
  for (const [key, value] of Object.entries(previous)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
});
function enable(): void {
  process.env.VERICLAIM_PRIVATE_MODE = "true";
  process.env.VERICLAIM_OWNER_USER = "owner";
  process.env.VERICLAIM_OWNER_PASSWORD = "synthetic-test-owner-password-long-enough";
  process.env.VERICLAIM_API_ACCESS_TOKEN = "synthetic-test-backend-bearer-key-long-enough";
}
test("private mode fails closed without credentials", () => {
  delete process.env.VERICLAIM_OWNER_PASSWORD;
  process.env.VERICLAIM_PRIVATE_MODE = "true";
  assert.equal(privateModeEnabled(), true);
  assert.equal(privateModeConfigured(), false);
  assert.equal(ownerAuthorized(null), false);
});
test("private mode verifies only exact owner credentials", () => {
  enable();
  assert.equal(privateModeConfigured(), true);
  const header = `Basic ${Buffer.from("owner:synthetic-test-owner-password-long-enough").toString("base64")}`;
  assert.equal(ownerAuthorized(header), true);
  for (const item of [null, "", "Bearer test", "Basic !!", `Basic ${Buffer.from("owner:incorrect").toString("base64")}`, `Basic ${Buffer.from("attacker:synthetic-test-owner-password-long-enough").toString("base64")}`]) {
    assert.equal(ownerAuthorized(item), false);
  }
});
test("not private means no owner authentication even with supplied credentials", () => {
  enable();
  process.env.VERICLAIM_PRIVATE_MODE = "false";
  assert.equal(ownerAuthorized(`Basic ${Buffer.from("owner:synthetic-test-owner-password-long-enough").toString("base64")}`), false);
});
