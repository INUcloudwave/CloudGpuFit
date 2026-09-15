"use client";

import { useEffect, useRef, useState } from "react";
import type { HfSearchResult, ModelConfigFields, ModelTypeErrorResponse, ModelTypeResponse, WorkloadType } from "@/lib/types";
import { formatCompactNumber } from "@/lib/format";

const DEBOUNCE_MS = 300;

interface Props {
  onClassified: (modelId: string, type: WorkloadType, config: ModelConfigFields) => void;
}

function classifyErrorMessage(status: number, body: ModelTypeErrorResponse | null): string {
  if (status === 404 || body?.error === "not_found") {
    return "이 저장소에는 config.json이 없습니다. transformers 호환 모델을 선택해주세요.";
  }
  if (status === 401 || status === 403 || body?.error === "gated") {
    return "접근 제한된 모델입니다. 다른 모델을 선택해주세요.";
  }
  return "모델 유형을 확인하는 중 오류가 발생했습니다. 다시 시도해주세요.";
}

export default function ModelSearchStep({ onClassified }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<HfSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [highlighted, setHighlighted] = useState(0);
  const [selected, setSelected] = useState<HfSearchResult | null>(null);
  const [checking, setChecking] = useState(false);
  const [classifyError, setClassifyError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (selected || !query.trim()) {
      return;
    }

    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setSearching(true);
      setSearchError(null);

      fetch(`/api/hf/search?q=${encodeURIComponent(query.trim())}`, { signal: controller.signal })
        .then(async (res) => {
          const data = await res.json();
          if (!res.ok || (data && "error" in data)) {
            setResults([]);
            setSearchError("검색 중 오류가 발생했습니다.");
            return;
          }
          setResults(data as HfSearchResult[]);
          setHighlighted(0);
        })
        .catch((err) => {
          if (err instanceof Error && err.name === "AbortError") return;
          setResults([]);
          setSearchError("검색 중 오류가 발생했습니다.");
        })
        .finally(() => setSearching(false));
    }, DEBOUNCE_MS);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query, selected]);

  const dropdownOpen = !selected && query.trim().length > 0 && results.length > 0;
  const showSearchError = !!searchError && !selected && query.trim().length > 0;

  function selectModel(model: HfSearchResult) {
    setSelected(model);
    setQuery(model.id);
    setResults([]);
    setClassifyError(null);
  }

  function handleQueryChange(value: string) {
    setQuery(value);
    setSelected(null);
    setClassifyError(null);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!dropdownOpen) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlighted((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlighted((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const model = results[highlighted];
      if (model) selectModel(model);
    } else if (e.key === "Escape") {
      setResults([]);
    }
  }

  async function handleNext() {
    if (!selected) return;
    setChecking(true);
    setClassifyError(null);
    try {
      const res = await fetch(`/api/hf/model-type?id=${encodeURIComponent(selected.id)}`);
      const data = await res.json();

      if (!res.ok) {
        setClassifyError(classifyErrorMessage(res.status, data as ModelTypeErrorResponse));
        return;
      }

      const body = data as ModelTypeResponse;
      if (body.type === "unsupported") {
        setClassifyError(
          "현재 지원하는 4개 유형(Decoder-only LLM / Vision classification / Text Enc-Dec / Audio Enc-Dec)에 해당하지 않습니다."
        );
        return;
      }

      onClassified(body.modelId, body.type, body.config);
    } catch {
      setClassifyError("모델 유형을 확인하는 중 오류가 발생했습니다. 다시 시도해주세요.");
    } finally {
      setChecking(false);
    }
  }

  return (
    <div className="relative mx-auto w-full max-w-xl">
      <label htmlFor="model-search" className="mb-2 block text-sm font-medium text-neutral-700 dark:text-neutral-300">
        Hugging Face 모델 검색
      </label>
      <div className="relative">
        <input
          id="model-search"
          type="text"
          value={query}
          placeholder="예: Qwen/Qwen3-0.6B"
          autoComplete="off"
          onChange={(e) => handleQueryChange(e.target.value)}
          onKeyDown={handleKeyDown}
          className="w-full rounded-lg border border-neutral-300 bg-white px-4 py-2.5 text-sm outline-none focus:border-neutral-500 focus:ring-2 focus:ring-neutral-200 dark:border-neutral-700 dark:bg-neutral-900 dark:focus:ring-neutral-700"
        />
        {searching && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-neutral-400">검색 중…</span>
        )}

        {dropdownOpen && (
          <ul className="absolute z-10 mt-1 w-full overflow-hidden rounded-lg border border-neutral-200 bg-white shadow-lg dark:border-neutral-700 dark:bg-neutral-900">
            {results.map((model, i) => (
              <li key={model.id}>
                <button
                  type="button"
                  onMouseEnter={() => setHighlighted(i)}
                  onClick={() => selectModel(model)}
                  className={`flex w-full items-center justify-between gap-3 px-4 py-2.5 text-left text-sm ${
                    i === highlighted ? "bg-neutral-100 dark:bg-neutral-800" : ""
                  }`}
                >
                  <span className="truncate font-mono">{model.id}</span>
                  <span className="flex shrink-0 items-center gap-2 text-xs text-neutral-500">
                    {model.pipeline_tag && (
                      <span className="rounded bg-neutral-100 px-1.5 py-0.5 dark:bg-neutral-800">
                        {model.pipeline_tag}
                      </span>
                    )}
                    <span>↓ {formatCompactNumber(model.downloads)}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {showSearchError && <p className="mt-2 text-sm text-red-600">{searchError}</p>}
      {classifyError && <p className="mt-2 text-sm text-red-600">{classifyError}</p>}

      <div className="mt-6 flex justify-end">
        <button
          type="button"
          disabled={!selected || checking}
          onClick={handleNext}
          className="rounded-lg bg-neutral-900 px-5 py-2.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40 dark:bg-white dark:text-neutral-900"
        >
          다음
        </button>
      </div>

      {checking && (
        <div className="fixed inset-0 z-20 flex items-center justify-center bg-black/40">
          <div className="flex flex-col items-center gap-3 rounded-xl bg-white px-8 py-6 dark:bg-neutral-900">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-neutral-300 border-t-neutral-800 dark:border-neutral-700 dark:border-t-white" />
            <p className="text-sm text-neutral-700 dark:text-neutral-300">모델 유형 확인 중…</p>
          </div>
        </div>
      )}
    </div>
  );
}
