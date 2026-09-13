"use client";

import { useCallback, useEffect, useState } from "react";
import { getMeta, type ApiResult } from "./api";
import type { Meta } from "./types";

export type MetaState = { kind: "loading" } | ApiResult<Meta>;

// /meta is fetched once per page load. Every figure on the page comes from it.
export function useMeta(): [MetaState, () => void] {
  const [state, setState] = useState<MetaState>({ kind: "loading" });

  const load = useCallback(() => {
    setState({ kind: "loading" });
    getMeta().then(setState);
  }, []);

  useEffect(() => {
    let live = true;
    getMeta().then((r) => {
      if (live) setState(r);
    });
    return () => {
      live = false;
    };
  }, []);

  return [state, load];
}
