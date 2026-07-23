"use client";

import {
  Fragment,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import type {
  Adu,
  Boundary,
  CorpusDocument,
  CuratedCase,
  EnglishAdu,
  ExplorerData,
  ModelKey,
  ViewKey,
} from "./types";


const MODEL_META: Record<
  ModelKey,
  { short: string; name: string; className: string; letter: string }
> = {
  eduseg_document: {
    short: "EduSeg · doc",
    name: "EduSeg · document context",
    className: "model-doc",
    letter: "D",
  },
  eduseg_adu: {
    short: "EduSeg · ADU",
    name: "EduSeg · per-ADU context",
    className: "model-adu",
    letter: "A",
  },
  secorel: {
    short: "SeCoRel",
    name: "SeCoRel · sentence chunks",
    className: "model-secorel",
    letter: "S",
  },
};

const FILTERS = [
  { value: "all", label: "All documents" },
  { value: "disagreement", label: "Model disagreement" },
  { value: "all-three", label: "All three agree" },
  { value: "missed-adu", label: "Missed ADU start" },
  { value: "english-split", label: "English ADU is split" },
  { value: "same-unit", label: "Same-Unit alternative" },
] as const;

type FilterKey = (typeof FILTERS)[number]["value"];
type SortKey = "review" | "id" | "proposals";


function ArrowIcon({ direction = "right" }: { direction?: "left" | "right" }) {
  return (
    <svg
      aria-hidden="true"
      className={direction === "left" ? "icon flip" : "icon"}
      viewBox="0 0 20 20"
    >
      <path d="M4 10h11M11 5l5 5-5 5" />
    </svg>
  );
}


function SearchIcon() {
  return (
    <svg aria-hidden="true" className="icon" viewBox="0 0 20 20">
      <circle cx="8.5" cy="8.5" r="5.5" />
      <path d="m13 13 4 4" />
    </svg>
  );
}


function BookIcon() {
  return (
    <svg aria-hidden="true" className="icon" viewBox="0 0 20 20">
      <path d="M3 3.5h5.5A2.5 2.5 0 0 1 11 6v11a2.5 2.5 0 0 0-2.5-2.5H3z" />
      <path d="M17 3.5h-3.5A2.5 2.5 0 0 0 11 6v11a2.5 2.5 0 0 1 2.5-2.5H17z" />
    </svg>
  );
}


function scrollToId(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}


function scoreLabel(score: number | null): string {
  return score === null ? "n/a" : score.toFixed(3);
}


function activeModels(view: ViewKey): ModelKey[] {
  return view === "compare"
    ? ["eduseg_document", "eduseg_adu", "secorel"]
    : [view];
}


function boundaryHasActivePrediction(boundary: Boundary, view: ViewKey): boolean {
  return activeModels(view).some((key) => boundary.layers[key].predicted);
}


function StatCard({
  value,
  label,
  detail,
  accent,
}: {
  value: number | string;
  label: string;
  detail: string;
  accent?: boolean;
}) {
  return (
    <div className={accent ? "stat-card stat-card-accent" : "stat-card"}>
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
      <span className="stat-detail">{detail}</span>
    </div>
  );
}


function AgreementBar({
  title,
  subtitle,
  values,
  f1,
}: {
  title: string;
  subtitle: string;
  values: { label: string; value: number; className: string }[];
  f1: number;
}) {
  const total = values.reduce((sum, value) => sum + value.value, 0);
  return (
    <article className="agreement-card">
      <div className="agreement-heading">
        <div>
          <p className="eyebrow">{subtitle}</p>
          <h3>{title}</h3>
        </div>
        <div className="f1-badge">
          <strong>{f1.toFixed(3)}</strong>
          <span>F1 agreement</span>
        </div>
      </div>
      <div className="stacked-bar" aria-label={`${title}: ${total} union boundaries`}>
        {values.map((value) => (
          <span
            className={value.className}
            key={value.label}
            style={{ width: `${(value.value / total) * 100}%` }}
            title={`${value.label}: ${value.value}`}
          />
        ))}
      </div>
      <div className="bar-legend">
        {values.map((value) => (
          <span key={value.label}>
            <i className={value.className} />
            <strong>{value.value}</strong> {value.label}
          </span>
        ))}
      </div>
      <p className="union-note">{total} distinct positions in the union</p>
    </article>
  );
}


function ModelDots({
  boundary,
  models,
}: {
  boundary: Boundary;
  models: ModelKey[];
}) {
  return (
    <span className="model-dots" aria-hidden="true">
      {models
        .filter((key) => boundary.layers[key].predicted)
        .map((key) => (
          <i className={MODEL_META[key].className} key={key}>
            {MODEL_META[key].letter}
          </i>
        ))}
    </span>
  );
}


function BoundaryMarker({
  boundary,
  view,
  selected,
  onSelect,
}: {
  boundary: Boundary;
  view: ViewKey;
  selected: boolean;
  onSelect: (boundary: Boundary) => void;
}) {
  const models = activeModels(view);
  const predicted = models.filter((key) => boundary.layers[key].predicted);
  const label = predicted.map((key) => MODEL_META[key].name).join(", ");
  return (
    <button
      aria-label={`Boundary at character ${boundary.offset}, predicted by ${label}`}
      className={selected ? "boundary-marker selected" : "boundary-marker"}
      data-boundary={boundary.offset}
      onClick={() => onSelect(boundary)}
      title={`Character ${boundary.offset} · ${label}`}
      type="button"
    >
      <span className="marker-line" />
      <ModelDots boundary={boundary} models={models} />
    </button>
  );
}


function GermanAduText({
  adu,
  document,
  view,
  selectedOffset,
  onSelect,
}: {
  adu: Adu;
  document: CorpusDocument;
  view: ViewKey;
  selectedOffset: number | null;
  onSelect: (boundary: Boundary) => void;
}) {
  const internalBoundaries = document.boundaries.filter(
    (boundary) =>
      !boundary.gold &&
      adu.start < boundary.offset &&
      boundary.offset < adu.end &&
      boundaryHasActivePrediction(boundary, view),
  );
  const parts: ReactNode[] = [];
  let cursor = adu.start;
  for (const boundary of internalBoundaries) {
    parts.push(
      <span className="text-chunk" key={`text-${cursor}`}>
        {document.german.text.slice(cursor, boundary.offset)}
      </span>,
    );
    parts.push(
      <BoundaryMarker
        boundary={boundary}
        key={`boundary-${boundary.offset}`}
        onSelect={onSelect}
        selected={selectedOffset === boundary.offset}
        view={view}
      />,
    );
    cursor = boundary.offset;
  }
  parts.push(
    <span className="text-chunk" key={`text-${cursor}`}>
      {document.german.text.slice(cursor, adu.end)}
    </span>,
  );
  return <div className="segmented-text" lang="de">{parts}</div>;
}


function EnglishAduText({ adu }: { adu: EnglishAdu }) {
  return (
    <div className="english-segmented-text" lang="en">
      {adu.edus.map((edu, index) => (
        <Fragment key={edu.id}>
          {index > 0 && (
            <span
              aria-label={`English gold EDU boundary before ${edu.id}`}
              className="english-gold-break"
              title={`English RST gold · ${edu.id}`}
            >
              <i />
              <small>EDU</small>
            </span>
          )}
          <span>{edu.text}</span>
          {index < adu.edus.length - 1 ? " " : ""}
        </Fragment>
      ))}
    </div>
  );
}


function LayerReadout({
  boundary,
  modelKey,
}: {
  boundary: Boundary;
  modelKey: ModelKey;
}) {
  const layer = boundary.layers[modelKey];
  return (
    <div className="layer-readout">
      <i className={MODEL_META[modelKey].className}>
        {MODEL_META[modelKey].letter}
      </i>
      <span>
        <strong>{MODEL_META[modelKey].short}</strong>
        <small>
          {layer.predicted
            ? "boundary"
            : layer.evidence === "not-comparable"
              ? "structural start"
              : "below threshold"}
        </small>
      </span>
      <code>{scoreLabel(layer.score)}</code>
    </div>
  );
}


function BoundaryInspector({
  boundary,
  onClose,
}: {
  boundary: Boundary;
  onClose: () => void;
}) {
  return (
    <aside className="boundary-inspector" aria-live="polite">
      <div className="inspector-heading">
        <div>
          <p className="eyebrow">
            {boundary.gold ? "Locked source boundary" : "Automatic proposal"}
          </p>
          <h3>
            {boundary.aduId} · character {boundary.offset}
          </h3>
        </div>
        <button aria-label="Close boundary details" onClick={onClose} type="button">
          ×
        </button>
      </div>
      <p className="context-quote">
        <span>{boundary.context.left}</span>
        <mark>|</mark>
        <span>{boundary.context.right}</span>
      </p>
      <div className="layer-grid">
        {(Object.keys(MODEL_META) as ModelKey[]).map((modelKey) => (
          <LayerReadout boundary={boundary} key={modelKey} modelKey={modelKey} />
        ))}
      </div>
      <p className="inspector-note">
        Scores are raw softmax boundary probabilities, not calibrated correctness
        estimates.
      </p>
    </aside>
  );
}


function DocumentCard({
  document,
  view,
  selectedOffset,
  showEnglish,
  onSelectBoundary,
}: {
  document: CorpusDocument;
  view: ViewKey;
  selectedOffset: number | null;
  showEnglish: boolean;
  onSelectBoundary: (boundary: Boundary) => void;
}) {
  return (
    <div className={showEnglish ? "adu-stack bilingual" : "adu-stack"}>
      {document.german.adus.map((adu, index) => {
        const englishAdu = document.english.adus[index];
        const startBoundary = document.boundaries.find(
          (boundary) => boundary.gold && boundary.offset === adu.start,
        );
        const missedBy = startBoundary
          ? (["eduseg_document", "secorel"] as ModelKey[]).filter(
              (key) =>
                adu.start > 0 &&
                startBoundary.layers[key].evidence === "below-threshold",
            )
          : [];
        return (
          <article className="adu-row" key={adu.id}>
            <div className="german-adu">
              <header>
                <button
                  className={
                    selectedOffset === adu.start
                      ? "adu-id-button selected"
                      : "adu-id-button"
                  }
                  data-boundary={adu.start}
                  onClick={() => startBoundary && onSelectBoundary(startBoundary)}
                  title={`Locked ADU start at character ${adu.start}`}
                  type="button"
                >
                  <span>{adu.id}</span>
                  <small>locked ADU</small>
                </button>
                {missedBy.length > 0 && (
                  <span className="missed-badge">
                    missed by {missedBy.map((key) => MODEL_META[key].letter).join(" + ")}
                  </span>
                )}
              </header>
              <GermanAduText
                adu={adu}
                document={document}
                onSelect={onSelectBoundary}
                selectedOffset={selectedOffset}
                view={view}
              />
            </div>
            {showEnglish && (
              <div className="english-adu">
                <header>
                  <span>aligned English</span>
                  <small>
                    {englishAdu.edus.length} gold EDU
                    {englishAdu.edus.length === 1 ? "" : "s"}
                  </small>
                </header>
                <EnglishAduText adu={englishAdu} />
              </div>
            )}
          </article>
        );
      })}
    </div>
  );
}


function CaseCard({
  item,
  onOpen,
}: {
  item: CuratedCase;
  onOpen: (item: CuratedCase) => void;
}) {
  return (
    <button className={`case-card ${item.kind}`} onClick={() => onOpen(item)} type="button">
      <span className="case-number">
        {item.docId.replace("micro_", "")} / {item.aduId}
      </span>
      <p className="eyebrow">{item.eyebrow}</p>
      <h3>{item.title}</h3>
      <p className="case-context">
        {item.context.left}
        <mark>|</mark>
        {item.context.right}
      </p>
      <p>{item.note}</p>
      <span className="case-link">
        Inspect boundary <ArrowIcon />
      </span>
    </button>
  );
}


function LoadingState({ error }: { error?: string }) {
  return (
    <main className="loading-screen">
      <div className="loading-mark">EDU</div>
      <h1>{error ? "The explorer data could not be loaded." : "Preparing the corpus…"}</h1>
      <p>
        {error ??
          "Loading 112 bilingual microtexts and their published boundary layers."}
      </p>
    </main>
  );
}


export function EduExplorer() {
  const [data, setData] = useState<ExplorerData | null>(null);
  const [error, setError] = useState<string | undefined>();
  const [selectedId, setSelectedId] = useState("micro_b001");
  const [view, setView] = useState<ViewKey>("compare");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<FilterKey>("all");
  const [sort, setSort] = useState<SortKey>("review");
  const [showEnglish, setShowEnglish] = useState(true);
  const [selectedOffset, setSelectedOffset] = useState<number | null>(405);
  const explorerRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const doc = params.get("doc");
    const requestedView = params.get("view") as ViewKey | null;
    if (doc) setSelectedId(doc);
    if (
      requestedView &&
      ["compare", "eduseg_document", "eduseg_adu", "secorel"].includes(requestedView)
    ) {
      setView(requestedView);
    }
    fetch("/data/edu-explorer.json")
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<ExplorerData>;
      })
      .then((value) => {
        setData(value);
        if (doc && !value.documents.some((item) => item.id === doc)) {
          setSelectedId(value.documents[0].id);
        }
      })
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : String(reason));
      });
  }, []);

  useEffect(() => {
    if (!data) return;
    const params = new URLSearchParams(window.location.search);
    params.set("doc", selectedId);
    params.set("view", view);
    window.history.replaceState(
      null,
      "",
      `?${params.toString()}${window.location.hash}`,
    );
  }, [data, selectedId, view]);

  useEffect(() => {
    if (!data || !window.location.hash) return;
    const targetId = window.location.hash.slice(1);
    const timer = window.setTimeout(() => {
      document.getElementById(targetId)?.scrollIntoView({ block: "start" });
    }, 180);
    return () => window.clearTimeout(timer);
  }, [data]);

  const filteredDocuments = useMemo(() => {
    if (!data) return [];
    const needle = query.trim().toLocaleLowerCase();
    const matching = data.documents.filter((document) => {
      if (
        needle &&
        !document.id.toLocaleLowerCase().includes(needle) &&
        !document.german.text.toLocaleLowerCase().includes(needle) &&
        !document.english.text.toLocaleLowerCase().includes(needle)
      ) {
        return false;
      }
      if (filter === "disagreement") return document.stats.disagreements > 0;
      if (filter === "all-three") return document.stats.allThreeShared > 0;
      if (filter === "missed-adu") {
        return (
          document.stats.missedAduEdUseg.length > 0 ||
          document.stats.missedAduSecorel.length > 0
        );
      }
      if (filter === "english-split") return document.stats.englishInternalGold > 0;
      if (filter === "same-unit") return document.sameUnitAffected;
      return true;
    });
    return matching.sort((a, b) => {
      if (sort === "id") return a.id.localeCompare(b.id, undefined, { numeric: true });
      if (sort === "proposals") {
        return (
          b.stats.edusegDocument +
          b.stats.secorel -
          a.stats.edusegDocument -
          a.stats.secorel ||
          a.id.localeCompare(b.id, undefined, { numeric: true })
        );
      }
      return (
        b.stats.reviewScore - a.stats.reviewScore ||
        a.id.localeCompare(b.id, undefined, { numeric: true })
      );
    });
  }, [data, filter, query, sort]);

  if (error) return <LoadingState error={error} />;
  if (!data) return <LoadingState />;

  const selectedDocument =
    data.documents.find((document) => document.id === selectedId) ?? data.documents[0];
  const selectedBoundary =
    selectedOffset === null
      ? undefined
      : selectedDocument.boundaries.find(
          (boundary) => boundary.offset === selectedOffset,
        );
  const selectedIndex = filteredDocuments.findIndex(
    (document) => document.id === selectedDocument.id,
  );

  function selectDocument(id: string) {
    setSelectedId(id);
    setSelectedOffset(null);
  }

  function openCase(item: CuratedCase) {
    setSelectedId(item.docId);
    setSelectedOffset(item.offset);
    setView("compare");
    setFilter("all");
    requestAnimationFrame(() => {
      explorerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      window.setTimeout(() => {
        document
          .querySelector(`[data-boundary="${item.offset}"]`)
          ?.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 550);
    });
  }

  function selectBoundary(boundary: Boundary) {
    setSelectedOffset(boundary.offset);
  }

  function stepDocument(delta: number) {
    if (filteredDocuments.length === 0) return;
    const current = selectedIndex >= 0 ? selectedIndex : 0;
    const next = (current + delta + filteredDocuments.length) % filteredDocuments.length;
    selectDocument(filteredDocuments[next].id);
  }

  const contextAgreement = data.summary.agreements.edusegContexts;
  const modelAgreement = data.summary.agreements.edusegSecorel;
  const corpus = data.summary.corpus;

  return (
    <>
      <header className="site-header">
        <a className="brand" href="#top" onClick={() => scrollToId("top")}>
          <span className="brand-mark">AM</span>
          <span>
            <strong>EDU Explorer</strong>
            <small>Arg-Microtexts research fork</small>
          </span>
        </a>
        <nav aria-label="Primary navigation">
          <button onClick={() => scrollToId("overview")} type="button">
            Findings
          </button>
          <button onClick={() => scrollToId("explorer")} type="button">
            Document explorer
          </button>
          <a
            href="https://github.com/mkrupo/arg-microtexts"
            rel="noreferrer"
            target="_blank"
          >
            Source ↗
          </a>
        </nav>
        <span className="status-chip">
          <i />
          published runs
        </span>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">German EDU segmentation · pre-gold analysis</p>
            <h1>
              Where do the
              <br />
              models draw the line?
            </h1>
            <p className="hero-lede">
              Compare German-only EduSeg, its context ablation, and multilingual
              SeCoRel—boundary by boundary, inside the original texts.
            </p>
            <div className="hero-actions">
              <button className="primary-action" onClick={() => scrollToId("explorer")} type="button">
                Explore all 112 texts <ArrowIcon />
              </button>
              <button className="text-action" onClick={() => scrollToId("overview")} type="button">
                Read the findings
              </button>
            </div>
          </div>
          <div className="hero-visual" aria-label="Example EDU segmentation">
            <div className="visual-label">
              <span>micro_b001 / a5</span>
              <small>three-model agreement · likely error</small>
            </div>
            <p lang="de">
              Wir Berliner sollten die Chance nutzen
              <span className="hero-boundary">
                <i className="model-doc">D</i>
                <i className="model-adu">A</i>
                <i className="model-secorel">S</i>
              </span>
              und Vorreiter im Mülltrennen werden!
            </p>
            <div className="visual-callout">
              <strong>Agreement ≠ gold</strong>
              <span>
                The shared subject and modal favor one EDU in our current working
                analysis.
              </span>
            </div>
          </div>
        </section>

        <section className="principle-strip" aria-label="Interpretation note">
          <span>01</span>
          <strong>Known structure</strong>
          <p>
            The 576 German ADU starts are locked source boundaries. The aligned English
            RST layer supplies 680 existing gold EDUs.
          </p>
          <span>02</span>
          <strong>What is still unknown</strong>
          <p>
            The models add German within-ADU candidates. Until human adjudication, they
            are proposals—not true or false EDUs.
          </p>
        </section>

        <section className="overview section-shell" id="overview">
          <div className="section-heading">
            <div>
              <p className="eyebrow">What we have</p>
              <h2>A verified bilingual base, then three German views</h2>
            </div>
            <p>
              Counts below come directly from the committed audit and published run
              summaries. No evaluation is inferred from the coarse ADUs.
            </p>
          </div>

          <div className="stats-grid">
            <StatCard
              detail="original German + professional English"
              label="microtexts"
              value={corpus.documents}
            />
            <StatCard
              detail="aligned across both languages"
              label="locked ADUs"
              value={corpus.adus}
            />
            <StatCard
              accent
              detail={`${corpus.split_adus} ADUs refined`}
              label="English gold EDUs"
              value={corpus.english_edus}
            />
            <StatCard
              detail="inside the coarse English ADUs"
              label="gold refinements"
              value={`+${corpus.internal_boundaries}`}
            />
          </div>

          <div className="method-grid">
            <article className="method-intro">
              <p className="eyebrow">Experiment design</p>
              <h3>Same texts, distinct input conditions</h3>
              <p>
                The protocols are intentionally not collapsed into a single leaderboard.
                Input context and tokenization are part of what is being tested.
              </p>
              <div className="locked-key">
                <i />
                <span>
                  <strong>ADU-constrained exports</strong>
                  restore any locked start missed by a model.
                </span>
              </div>
            </article>
            <article className="method-card method-doc">
              <span className="method-letter">D</span>
              <p className="eyebrow">Primary condition</p>
              <h3>EduSeg · document</h3>
              <strong>{data.summary.models.edusegDocument.internalProposals}</strong>
              <span>internal proposals</span>
              <p>{data.summary.models.edusegDocument.protocol}</p>
              <small>
                461 / 464 non-initial ADU starts recovered
              </small>
            </article>
            <article className="method-card method-adu">
              <span className="method-letter">A</span>
              <p className="eyebrow">Context ablation</p>
              <h3>EduSeg · per ADU</h3>
              <strong>{data.summary.models.edusegAdu.internalProposals}</strong>
              <span>internal proposals</span>
              <p>{data.summary.models.edusegAdu.protocol}</p>
              <small>ADU-start scores are not compared</small>
            </article>
            <article className="method-card method-secorel">
              <span className="method-letter">S</span>
              <p className="eyebrow">Model comparison</p>
              <h3>SeCoRel</h3>
              <strong>{data.summary.models.secorel.internalProposals}</strong>
              <span>internal proposals</span>
              <p>{data.summary.models.secorel.protocol}</p>
              <small>
                461 / 464 non-initial ADU starts recovered
              </small>
            </article>
          </div>

          <div className="agreement-section">
            <div className="mini-heading">
              <p className="eyebrow">Exact-offset agreement</p>
              <h2>The systems mostly agree—and differ in useful ways</h2>
              <p>
                These are comparisons of boundary sets at exact raw-text character
                offsets. They measure stability and complementarity, not German
                segmentation accuracy.
              </p>
            </div>
            <div className="agreement-grid">
              <AgreementBar
                f1={contextAgreement.f1}
                subtitle="One model, two context windows"
                title="EduSeg document vs. per ADU"
                values={[
                  {
                    label: "shared",
                    value: contextAgreement.shared,
                    className: "bar-shared",
                  },
                  {
                    label: "document only",
                    value: contextAgreement.documentOnly,
                    className: "bar-doc",
                  },
                  {
                    label: "ADU only",
                    value: contextAgreement.aduOnly,
                    className: "bar-adu",
                  },
                ]}
              />
              <AgreementBar
                f1={modelAgreement.f1}
                subtitle="Two architectures and interfaces"
                title="EduSeg document vs. SeCoRel"
                values={[
                  {
                    label: "shared",
                    value: modelAgreement.shared,
                    className: "bar-shared",
                  },
                  {
                    label: "EduSeg only",
                    value: modelAgreement.edusegOnly,
                    className: "bar-doc",
                  },
                  {
                    label: "SeCoRel only",
                    value: modelAgreement.secorelOnly,
                    className: "bar-secorel",
                  },
                ]}
              />
            </div>
          </div>

          <div className="cases-section">
            <div className="mini-heading">
              <p className="eyebrow">Three cases to keep in mind</p>
              <h2>Agreement helps prioritize review. It cannot replace it.</h2>
            </div>
            <div className="case-grid">
              {data.summary.curatedCases.map((item) => (
                <CaseCard item={item} key={`${item.docId}-${item.offset}`} onOpen={openCase} />
              ))}
            </div>
          </div>
        </section>

        <section className="explorer section-shell" id="explorer" ref={explorerRef}>
          <div className="section-heading explorer-title">
            <div>
              <p className="eyebrow">Document explorer</p>
              <h2>Inspect every proposal in its source context</h2>
            </div>
            <p>
              Select a marker for scores and exact context. Switch on English to compare
              against the aligned RST gold segmentation—without treating translation
              boundaries as German labels.
            </p>
          </div>

          <div className="explorer-layout">
            <aside className="document-sidebar">
              <div className="search-field">
                <SearchIcon />
                <input
                  aria-label="Search documents and text"
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search ID or text…"
                  type="search"
                  value={query}
                />
              </div>
              <div className="sidebar-controls">
                <label>
                  <span>Show</span>
                  <select
                    onChange={(event) => setFilter(event.target.value as FilterKey)}
                    value={filter}
                  >
                    {FILTERS.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Sort</span>
                  <select
                    onChange={(event) => setSort(event.target.value as SortKey)}
                    value={sort}
                  >
                    <option value="review">Review priority</option>
                    <option value="proposals">Most proposals</option>
                    <option value="id">Document ID</option>
                  </select>
                </label>
              </div>
              <div className="document-count">
                <strong>{filteredDocuments.length}</strong>
                <span>of {data.documents.length} texts</span>
              </div>
              <div className="document-list" role="listbox" aria-label="Corpus documents">
                {filteredDocuments.map((document) => (
                  <button
                    aria-selected={document.id === selectedDocument.id}
                    className={
                      document.id === selectedDocument.id
                        ? "document-item selected"
                        : "document-item"
                    }
                    key={document.id}
                    onClick={() => selectDocument(document.id)}
                    role="option"
                    type="button"
                  >
                    <span>
                      <strong>{document.id.replace("micro_", "")}</strong>
                      <small>{document.stats.aduCount} ADUs</small>
                    </span>
                    <span className="document-item-stats">
                      {document.stats.disagreements > 0 && (
                        <i title={`${document.stats.disagreements} disagreements`}>
                          {document.stats.disagreements} Δ
                        </i>
                      )}
                      <b>{document.stats.edusegDocument}/{document.stats.secorel}</b>
                    </span>
                  </button>
                ))}
                {filteredDocuments.length === 0 && (
                  <p className="empty-list">No document matches these filters.</p>
                )}
              </div>
            </aside>

            <div className="document-workspace">
              <header className="document-heading">
                <div>
                  <p className="eyebrow">Original German microtext</p>
                  <h2>{selectedDocument.id}</h2>
                  <div className="document-tags">
                    <span>{selectedDocument.stats.aduCount} locked ADUs</span>
                    <span>{selectedDocument.stats.disagreements} disagreements</span>
                    <span>
                      {selectedDocument.stats.englishInternalGold} English refinements
                    </span>
                    {selectedDocument.sameUnitAffected && <span>Same-Unit alternative</span>}
                  </div>
                </div>
                <div className="document-nav">
                  <button
                    aria-label="Previous filtered document"
                    disabled={filteredDocuments.length === 0}
                    onClick={() => stepDocument(-1)}
                    type="button"
                  >
                    <ArrowIcon direction="left" />
                  </button>
                  <span>
                    {selectedIndex >= 0 ? selectedIndex + 1 : "—"} /{" "}
                    {filteredDocuments.length}
                  </span>
                  <button
                    aria-label="Next filtered document"
                    disabled={filteredDocuments.length === 0}
                    onClick={() => stepDocument(1)}
                    type="button"
                  >
                    <ArrowIcon />
                  </button>
                </div>
              </header>

              <div className="view-toolbar">
                <div className="view-tabs" aria-label="Boundary layer" role="group">
                  {(
                    [
                      ["compare", "Compare all"],
                      ["eduseg_document", "EduSeg · doc"],
                      ["eduseg_adu", "EduSeg · ADU"],
                      ["secorel", "SeCoRel"],
                    ] as [ViewKey, string][]
                  ).map(([key, label]) => (
                    <button
                      aria-pressed={view === key}
                      className={view === key ? "selected" : ""}
                      key={key}
                      onClick={() => setView(key)}
                      type="button"
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <label className="english-toggle">
                  <input
                    checked={showEnglish}
                    onChange={(event) => setShowEnglish(event.target.checked)}
                    type="checkbox"
                  />
                  <span />
                  <BookIcon />
                  English gold
                </label>
              </div>

              <div className="legend">
                <span><i className="locked-dot" /> ADU · locked</span>
                {activeModels(view).map((key) => (
                  <span key={key}>
                    <i className={MODEL_META[key].className} />
                    {MODEL_META[key].short}
                  </span>
                ))}
                {showEnglish && <span><i className="english-dot" /> English RST gold</span>}
                <small>Click a marker for evidence</small>
              </div>

              {selectedBoundary && (
                <BoundaryInspector
                  boundary={selectedBoundary}
                  onClose={() => setSelectedOffset(null)}
                />
              )}

              <DocumentCard
                document={selectedDocument}
                onSelectBoundary={selectBoundary}
                selectedOffset={selectedOffset}
                showEnglish={showEnglish}
                view={view}
              />

              <footer className="document-summary">
                <div>
                  <span>Within-ADU proposals</span>
                  <strong>
                    <i className="model-doc">D</i>
                    {selectedDocument.stats.edusegDocument}
                  </strong>
                  <strong>
                    <i className="model-adu">A</i>
                    {selectedDocument.stats.edusegAdu}
                  </strong>
                  <strong>
                    <i className="model-secorel">S</i>
                    {selectedDocument.stats.secorel}
                  </strong>
                </div>
                <p>
                  {selectedDocument.stats.allThreeShared} shared by all three ·{" "}
                  {selectedDocument.stats.disagreements} exact-offset disagreements
                </p>
              </footer>
            </div>
          </div>
        </section>

        <section className="provenance">
          <div>
            <p className="eyebrow">How to read this site</p>
            <h2>The interface is a view over committed research artifacts</h2>
          </div>
          <ol>
            <li>
              <span>1</span>
              <p>
                Original raw text supplies every displayed character and canonical
                offset.
              </p>
            </li>
            <li>
              <span>2</span>
              <p>
                Gold ADU starts and English RST EDUs come from audited source layers.
              </p>
            </li>
            <li>
              <span>3</span>
              <p>
                Automatic boundaries and scores come from frozen, published run tables.
              </p>
            </li>
          </ol>
          <p className="provenance-note">
            Softmax scores are shown for inspection only. “Shared” always means exact
            character-offset agreement. Human German re-annotation and adjudication are
            the next validity step.
          </p>
        </section>
      </main>

      <footer className="site-footer">
        <span className="brand-mark">AM</span>
        <p>
          Arg-Microtexts EDU Explorer
          <small>Research fork · automatic German proposals are not gold</small>
        </p>
        <a
          href="https://github.com/mkrupo/arg-microtexts/blob/master/docs/EXPERIMENTS.md"
          rel="noreferrer"
          target="_blank"
        >
          Experiment documentation <ArrowIcon />
        </a>
      </footer>
    </>
  );
}
