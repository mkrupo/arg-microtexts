export type ModelKey = "eduseg_document" | "eduseg_adu" | "secorel";
export type ViewKey = "compare" | ModelKey;

export type LayerEvidence = {
  predicted: boolean;
  score: number | null;
  evidence: "predicted" | "below-threshold" | "not-comparable";
};

export type Boundary = {
  offset: number;
  aduId: string;
  gold: boolean;
  boundaryClass: "document-start" | "adu" | "internal";
  afterTerminalPunctuation: boolean;
  context: { left: string; right: string };
  layers: Record<ModelKey, LayerEvidence>;
};

export type Adu = {
  id: string;
  sourceUnitId: string;
  start: number;
  end: number;
  text: string;
};

export type EnglishEdu = {
  id: string;
  start: number;
  end: number;
  text: string;
  internal: boolean;
};

export type EnglishAdu = Adu & {
  edus: EnglishEdu[];
};

export type DocumentStats = {
  aduCount: number;
  englishEduCount: number;
  englishInternalGold: number;
  edusegDocument: number;
  edusegAdu: number;
  secorel: number;
  edusegSecorelShared: number;
  edusegSecorelUnion: number;
  allThreeShared: number;
  disagreements: number;
  missedAduEdUseg: number[];
  missedAduSecorel: number[];
  reviewScore: number;
};

export type CorpusDocument = {
  id: string;
  group: string;
  sameUnitAffected: boolean;
  german: { text: string; adus: Adu[] };
  english: { text: string; adus: EnglishAdu[] };
  boundaries: Boundary[];
  stats: DocumentStats;
};

export type CuratedCase = {
  docId: string;
  aduId: string;
  offset: number;
  kind: string;
  eyebrow: string;
  title: string;
  note: string;
  context: { left: string; right: string };
  layers: Record<ModelKey, LayerEvidence>;
};

export type ExplorerData = {
  schemaVersion: number;
  generatedFrom: {
    originalCorpusTree: string;
    multilayerCommit: string;
    runProjectCommits: Record<ModelKey, string>;
    sources: string[];
  };
  summary: {
    corpus: {
      documents: number;
      adus: number;
      english_edus: number;
      split_adus: number;
      internal_boundaries: number;
      refined_documents: number;
      sameunit_documents: number;
    };
    models: {
      edusegDocument: {
        label: string;
        protocol: string;
        internalProposals: number;
        recoveredAduStarts: number;
        goldAduStarts: number;
      };
      edusegAdu: {
        label: string;
        protocol: string;
        internalProposals: number;
      };
      secorel: {
        label: string;
        protocol: string;
        internalProposals: number;
        recoveredAduStarts: number;
        goldAduStarts: number;
      };
    };
    agreements: {
      edusegContexts: {
        shared: number;
        documentOnly: number;
        aduOnly: number;
        f1: number;
        jaccard: number;
      };
      edusegSecorel: {
        shared: number;
        edusegOnly: number;
        secorelOnly: number;
        f1: number;
        jaccard: number;
      };
    };
    groups: Record<
      string,
      {
        documents: number;
        edusegDocument: number;
        edusegAdu: number;
        secorel: number;
      }
    >;
    curatedCases: CuratedCase[];
  };
  documents: CorpusDocument[];
};
