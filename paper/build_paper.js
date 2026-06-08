const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak,
} = require("docx");

// ---------------------------------------------------------------------------
// 공통 상수 및 헬퍼
// ---------------------------------------------------------------------------
const FONT = "Malgun Gothic";          // 한글/라틴 모두 지원하는 크로스플랫폼 폰트
const BODY = 20;                        // 10pt (half-points)
const CONTENT_W = 9026;                 // A4, 1인치 여백 기준 본문 폭(DXA)
const LINE = { line: 336, lineRule: "auto" }; // 약 1.4배 줄간격

// 본문 단락 (양쪽 정렬)
function P(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 120, ...LINE },
    indent: opts.firstLine !== false ? { firstLine: 200 } : undefined,
    children: runs(text),
  });
}
// 들여쓰기 없는 단락
function PN(text, opts = {}) { return P(text, { firstLine: false, ...opts }); }

// 굵게/일반 혼합 런 파서: **bold** 구간을 굵게 처리
function runs(text) {
  const parts = String(text).split(/(\*\*[^*]+\*\*)/g).filter(s => s.length);
  return parts.map(s => {
    if (s.startsWith("**") && s.endsWith("**")) {
      return new TextRun({ text: s.slice(2, -2), bold: true });
    }
    return new TextRun({ text: s });
  });
}

function H1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 160 },
    children: [new TextRun(text)],
  });
}
function H2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 220, after: 120 },
    children: [new TextRun(text)],
  });
}
function H3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 160, after: 100 },
    children: [new TextRun(text)],
  });
}

function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 80, ...LINE },
    children: runs(text),
  });
}
function numbered(text, ref = "nums") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 80, ...LINE },
    children: runs(text),
  });
}

// 그림 캡션
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 200 },
    children: [new TextRun({ text, italics: true, size: 18, color: "444444" })],
  });
}
// 표 캡션
function tableCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 160, after: 80 },
    children: [new TextRun({ text, bold: true, size: 19 })],
  });
}

function img(path) {
  const w = 560, h = Math.round(560 * 0.5184);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 40 },
    children: [new ImageRun({
      type: "png",
      data: fs.readFileSync(path),
      transformation: { width: w, height: h },
      altText: { title: "결과 그래프", description: "실험 결과", name: "figure" },
    })],
  });
}

// ---- 표 헬퍼 -------------------------------------------------------------
const BC = "B0B0B0";
const cellBorder = {
  top: { style: BorderStyle.SINGLE, size: 1, color: BC },
  bottom: { style: BorderStyle.SINGLE, size: 1, color: BC },
  left: { style: BorderStyle.SINGLE, size: 1, color: BC },
  right: { style: BorderStyle.SINGLE, size: 1, color: BC },
};
function tcell(text, width, { head = false, fill, align = AlignmentType.LEFT, bold = false } = {}) {
  return new TableCell({
    borders: cellBorder,
    width: { size: width, type: WidthType.DXA },
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({
      alignment: align,
      spacing: { after: 0, line: 264, lineRule: "auto" },
      children: [new TextRun({ text, bold: head || bold, size: 18,
        color: head ? "FFFFFF" : "000000" })],
    })],
  });
}
function makeTable(widths, headers, rows, headFill = "365F91") {
  const headRow = new TableRow({
    tableHeader: true,
    children: headers.map((t, i) =>
      tcell(t, widths[i], { head: true, fill: headFill,
        align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })),
  });
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => {
      const isObj = typeof c === "object" && c !== null;
      const text = isObj ? c.t : c;
      return tcell(text, widths[i], {
        align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        fill: isObj && c.fill ? c.fill : (ri % 2 === 1 ? "F2F5F9" : undefined),
        bold: isObj ? !!c.bold : false,
      });
    }),
  }));
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [headRow, ...bodyRows],
  });
}

// 강조 박스(인용/시사점)
function calloutBox(label, text, fill = "EAF1FB", bar = "365F91") {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 120, after: 160, line: 312, lineRule: "auto" },
    shading: { fill, type: ShadingType.CLEAR },
    border: {
      left: { style: BorderStyle.SINGLE, size: 24, color: bar, space: 10 },
    },
    indent: { left: 200, right: 160 },
    children: [
      new TextRun({ text: label + "  ", bold: true, color: bar }),
      ...runs(text),
    ],
  });
}

const FIG = (f) => `/Users/hansol/Documents/Study/jinro-tamseok-project/paper/figures/${f}`;

// ===========================================================================
// 문서 본문 구성
// ===========================================================================
const children = [];

// ---- 표지/제목 ----
children.push(
  new Paragraph({ spacing: { before: 240, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "연합학습에서의 그래디언트 유출 공격과 방어의 공방 분석",
      bold: true, size: 36 })] }),
  new Paragraph({ spacing: { before: 80, after: 40 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "— 이미지 분류 및 그래프 신경망 모델을 중심으로 —",
      bold: true, size: 26, color: "333333" })] }),
  new Paragraph({ spacing: { before: 60, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text:
      "An Analysis of the Attack–Defense Arms Race in Gradient Leakage for Federated Learning:",
      italics: true, size: 19, color: "555555" })] }),
  new Paragraph({ spacing: { after: 220 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text:
      "A Study on Image Classification and Graph Neural Network Models",
      italics: true, size: 19, color: "555555" })] }),
  new Paragraph({ spacing: { after: 20 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "진로탐색 연구 프로젝트", size: 20, color: "333333" })] }),
  new Paragraph({ spacing: { after: 240 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "2026", size: 18, color: "777777" })] }),
);

// 구분선
children.push(new Paragraph({
  spacing: { after: 160 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "365F91", space: 1 } },
  children: [new TextRun("")],
}));

// ---- 국문 초록 ----
children.push(
  new Paragraph({ spacing: { before: 80, after: 100 },
    children: [new TextRun({ text: "국문 초록", bold: true, size: 24 })] }),
  PN("연합학습(Federated Learning)은 원본 데이터를 중앙 서버로 전송하지 않고 모델의 그래디언트(Gradient)만을 공유함으로써 프라이버시를 보존하도록 설계된 분산 기계학습 패러다임이다. 그러나 최근 연구들은 공유되는 그래디언트만으로도 원본 학습 데이터를 역으로 복원할 수 있는 그래디언트 유출(Gradient Leakage), 즉 DLG(Deep Leakage from Gradients) 공격의 위협을 보고하였다. 본 연구는 이러한 위협이 이미지(CIFAR-100)와 그래프(WikiCS, FedLoG) 두 도메인에서 어떻게 발현되는지, 그리고 다양한 방어가 적응적(adaptive) 공격 앞에서 어느 정도의 견고성을 갖는지를, **재현성과 통계적 엄밀성을 갖춘 다중 시드(n=5) 프로토콜** 하에서 실증 분석한다. 특히 본 연구는 초기 구현이 범했던 단일 실행(n=1) 보고의 위험성을 정면으로 다룬다. 시드를 고정하지 않은 원본 코드를 5회 재실행하자 그래프 노드 특성의 복원 코사인 유사도는 0.518에서 0.999까지(평균 0.85±0.22) 출렁였으며, 이는 단일 실행으로 보고된 ‘0.9999’가 대표값이 아님을 보여준다. 멀티시드 실험에서 노드별 절대 복원도는 분산이 매우 컸으나(예: baseline 0.384±0.316), 동일 타겟 노드를 공유하는 시드 내 쌍체(paired) 비교에서는 공방 효과가 일관되게 드러났다. 즉 특징 스케일링(Feature Scaling) 난독화는 신뢰할 만한 방어가 아니며(Δ +0.001), 비밀 스케일까지 함께 추정하는 조인트 최적화 공격이 이를 일관되게 무력화하였다(Δ +0.467, 전 시드 양수). 그 근본 원인은 복원 품질 지표인 코사인 유사도가 스케일에 불변이라는 점으로, 공격자가 비밀 스케일을 크게 오추정해도 특성 방향은 그대로 복원되었다. 반면 그래디언트 클리핑과 보정 노이즈를 결합한 차분 프라이버시(DP)는 적응적 공격조차 전 시드에서 일관되게 차단하였으며(복원 코사인 0.002±0.013), 극단적 희소화도 방어에 성공하였다(Δ −0.354). 이미지 도메인에서는 라벨이 모든 조건에서 100% 유출되었고, 단순 압축 기법은 실질적 방어 효과가 없었으며(복원 MSE는 대부분 무방어와 동일), Soteria는 그래디언트 충실도를 0.73으로 떨어뜨려 유용성만 희생하고 프라이버시 이득은 없었다. 본 연구는 난독화 기반 방어의 근본적 한계와 단일 실행 보고의 위험성을 드러내고, 정보 훼손형 방어(DP·극단 희소화)만이 적응적 공격자에 대해 일관된 보장을 제공하되 유용성 비용을 동반함을 시사한다."),
  new Paragraph({ spacing: { before: 80, after: 220 }, alignment: AlignmentType.JUSTIFIED,
    children: [
      new TextRun({ text: "주제어: ", bold: true }),
      new TextRun("연합학습, 그래디언트 유출, Deep Leakage from Gradients, 차분 프라이버시, 그래프 신경망, 적대적 공격, 재현성, 프라이버시-유용성 절충, 프라이버시 보존 기계학습"),
    ] }),
);

children.push(new Paragraph({ children: [new PageBreak()] }));

// ===========================================================================
// 1. 서론
// ===========================================================================
children.push(H1("1. 서론"));

children.push(H2("1.1 연구 배경"));
children.push(P("인공지능 모델의 성능은 학습에 사용되는 데이터의 양과 질에 크게 의존한다. 그러나 의료 기록, 금융 거래 내역, 소셜 네트워크 프로필과 같이 민감한 개인정보를 담은 데이터는 법적·윤리적 제약으로 인해 한곳에 모으기 어렵다. 이러한 배경에서 등장한 **연합학습(Federated Learning)**은 데이터를 각 참여자(클라이언트)의 기기에 그대로 둔 채, 모델 파라미터의 변화량인 그래디언트(또는 가중치 업데이트)만을 중앙 서버로 전송하여 글로벌 모델을 학습하는 분산 학습 방식이다. 데이터 원본이 기기를 떠나지 않으므로, 연합학습은 오랫동안 '프라이버시를 보존하는' 학습 방법으로 여겨져 왔다."));
children.push(P("그러나 Zhu 등(2019)이 제안한 **DLG(Deep Leakage from Gradients)** 공격은 이러한 통념에 정면으로 도전하였다. 이들은 클라이언트가 전송한 그래디언트만으로 원본 학습 이미지와 라벨을 거의 완벽하게 복원할 수 있음을 보였다. 핵심 아이디어는 단순하다. 공격자가 임의의 더미(dummy) 입력을 만든 뒤, 그 더미 입력이 만들어내는 그래디언트가 가로챈 실제 그래디언트와 같아지도록 더미 입력 자체를 최적화하면, 결국 더미 입력은 원본 데이터로 수렴한다. 이후 iDLG, Inverting Gradients(Geiping 등, 2020) 등 후속 연구가 복원 품질과 안정성을 크게 끌어올리면서, 그래디언트 유출은 연합학습의 핵심 위협으로 자리 잡았다."));

children.push(H2("1.2 문제 제기 및 연구 질문"));
children.push(P("기존 연구는 주로 이미지 분류 모델을 대상으로 그래디언트 유출을 다루어 왔다. 그러나 실제 산업 현장에서 연합학습은 이미지뿐 아니라 그래프 형태의 관계형 데이터—예컨대 소셜 그래프, 거래 네트워크, 분자 구조 등—에도 광범위하게 적용되고 있다. 그래프 신경망(GNN)의 연합학습에서 그래디언트 유출이 어떻게 작동하는지, 그리고 이미지와 비교해 그 위험도가 어떠한지는 상대적으로 덜 조명되었다."));
children.push(P("더 근본적인 문제는 방어와 공격 사이의 **공방(arms race)**이다. 어떤 방어가 특정 공격을 막았다고 해서, 그 방어의 존재와 원리를 파악한 '적응적 공격자'까지 막을 수 있는 것은 아니다. 방어 기법의 진정한 강건성은 그 메커니즘이 공격자에게 노출된 최악의 상황에서 검증되어야 한다. 이에 본 연구는 다음 세 가지 연구 질문(RQ)에 답하고자 한다."));
children.push(numbered("**RQ1.** 이미지 분류 모델에서 흔히 사용되는 그래디언트 압축 기법(희소화·양자화·가지치기·Soteria)은 DLG 공격에 대해 실질적인 방어 효과를 제공하는가?"));
children.push(numbered("**RQ2.** 그래프 신경망의 연합학습에서 노드 특성은 그래디언트만으로 얼마나 복원 가능하며, 그 위험도는 이미지 도메인과 비교해 어떠한가?"));
children.push(numbered("**RQ3.** 방어 메커니즘이 공격자에게 노출되었을 때, 난독화 기반 방어와 정보 훼손 기반 방어는 적응적 공격에 대해 각각 어떤 강건성을 보이는가?"));

children.push(H2("1.3 연구의 기여"));
children.push(P("본 연구의 기여는 다음과 같이 요약된다."));
children.push(bullet("**이중 도메인 실증 분석:** 동일한 위협 모델 하에서 이미지(CIFAR-100)와 그래프(WikiCS) 두 도메인의 그래디언트 유출을 통합적으로 구현하고 정량 비교하였다."));
children.push(bullet("**최신 GNN 모델에 대한 공격 검증:** ICLR 2025에서 발표된 서브그래프 연합학습 모델 FedLoG를 대상으로 노드 특성 복원 공격을 구현하여, 저차원 밀집 벡터의 취약성을 실증하였다."));
children.push(bullet("**공방 시나리오의 단계적 구성:** ‘기본 취약성 → 내장 방어 → 적응적 공격에 의한 방어 파훼 → 심화 방어’로 이어지는 4단계 공방을 설계하여, 난독화 방어의 한계와 훼손성 방어의 필요성을 명확히 드러냈다."));

children.push(H2("1.4 논문의 구성"));
children.push(P("본 논문의 나머지는 다음과 같이 구성된다. 2장에서는 연합학습, 그래디언트 유출 공격, 주요 방어 기법, 그리고 서브그래프 연합학습에 관한 이론적 배경과 관련 연구를 정리한다. 3장에서는 위협 모델과 전체 실험 설계 및 평가 지표를 기술한다. 4장은 이미지 분류 모델(트랙 I)에 대한 공격과 방어 실험을, 5장은 그래프 신경망(트랙 II)에 대한 공격–방어–적응적 공격–심화 방어의 공방 실험을 상세히 다룬다. 6장에서는 결과를 종합 고찰하고 연구의 한계를 논하며, 7장에서 결론을 맺는다."));

// ===========================================================================
// 2. 이론적 배경 및 관련 연구
// ===========================================================================
children.push(H1("2. 이론적 배경 및 관련 연구"));

children.push(H2("2.1 연합학습과 FedAvg"));
children.push(P("연합학습은 McMahan 등(2017)이 제안한 **FedAvg(Federated Averaging)** 알고리즘으로 대표된다. 각 라운드에서 서버는 글로벌 모델을 다수의 클라이언트에 배포하고, 각 클라이언트는 자신의 로컬 데이터로 모델을 학습한 뒤 그 결과(그래디언트 또는 갱신된 가중치)를 서버로 보낸다. 서버는 이를 가중 평균하여 글로벌 모델을 갱신한다. 데이터 원본은 결코 전송되지 않으므로, 통신 비용을 줄이는 동시에 데이터 지역성(locality)을 유지한다는 장점이 있다."));
children.push(P("본 연구에서는 이러한 FedAvg 절차를 직접 구현하여, 다수 클라이언트가 그래디언트를 계산·전송하고 서버가 이를 평균 집계하여 글로벌 모델을 갱신하는 과정을 시뮬레이션하였다. 중요한 점은, **악의적 서버**가 그래디언트를 평균 집계하기 직전에 특정 클라이언트의 그래디언트를 단독으로 분리·탈취할 수 있다는 구조적 취약성이다. 본 연구의 공격은 바로 이 지점을 파고든다."));

children.push(H2("2.2 그래디언트 유출 공격: DLG 계열"));
children.push(P("DLG(Zhu 등, 2019)의 핵심은 그래디언트 매칭을 통한 최적화이다. 공격자는 더미 입력 x′과 더미 라벨 y′을 무작위로 초기화하고, 이들로부터 계산한 더미 그래디언트 ∇′와 가로챈 실제 그래디언트 ∇ 사이의 거리를 최소화하도록 x′, y′을 갱신한다. 원래 DLG는 L2 거리와 L-BFGS 최적화를 사용했으나, 이는 발산이나 국소 최적값에 빠지기 쉬웠다."));
children.push(P("Geiping 등(2020)의 **Inverting Gradients**는 두 가지 핵심 개선을 제시했다. 첫째, 그래디언트의 크기(magnitude)에 불변인 **코사인 유사도(cosine similarity)** 손실을 사용하여 방향성만을 매칭함으로써 최적화를 안정화했다. 둘째, 자연 영상의 매끄러움을 유도하는 **전변동(Total Variation, TV) 정규화**를 추가하여 복원 이미지의 노이즈를 억제했다. 본 연구의 공격 알고리즘은 이 두 기법을 기반으로 하며, 그래프 도메인에서는 TV 대신 L2 정규화를 사용하도록 변형하였다."));

children.push(H2("2.3 그래디언트 압축 및 프라이버시 방어 기법"));
children.push(P("그래디언트 유출에 대한 방어는 크게 두 부류로 나뉜다. 첫째는 통신 효율을 위해 고안되었으나 부수적으로 방어 효과를 갖는 **압축 기법**이다."));
children.push(bullet("**희소화(Sparsification):** 절댓값이 작은 하위 일정 비율의 그래디언트 성분을 0으로 만든다. 전송량을 줄이는 동시에 정보를 제거한다."));
children.push(bullet("**양자화(Quantization):** 그래디언트의 수치 정밀도를 낮춘다(예: FP32→FP16). 통신량은 줄지만 방향 정보는 대체로 보존된다."));
children.push(bullet("**가지치기(Pruning):** DLG 원 논문이 제시한 방어로, 텐서별이 아니라 모델 전체 그래디언트를 기준으로 하위 비율을 일괄 마스킹한다."));
children.push(P("둘째는 프라이버시를 직접 겨냥해 설계된 **전용 방어 기법**이다."));
children.push(bullet("**Soteria(Sun 등, 2021):** 특정 레이어(주로 마지막 완전연결층)의 입력 표현(representation) 중요도를 분석하여, 데이터 유출에 결정적으로 기여하는 성분의 그래디언트만 선택적으로 0으로 만든다. 모델 정확도 손실을 최소화하면서 유출을 막는 것을 목표로 한다."));
children.push(bullet("**차분 프라이버시(Differential Privacy, DP):** Abadi 등(2016)의 DP-SGD처럼, 그래디언트에 보정된 가우시안 노이즈를 주입하여 개별 데이터의 기여를 수학적으로 가린다. 이론적 프라이버시 보장을 제공하는 대신 모델 유용성을 일부 희생한다."));

children.push(H2("2.4 그래프 신경망과 서브그래프 연합학습(FedLoG)"));
children.push(P("그래프 신경망(GNN)은 노드와 엣지로 구성된 그래프에서 이웃 정보를 집계하여 노드 표현을 학습한다. 본 연구가 대상으로 삼은 **GraphSAGE**(Hamilton 등, 2017)는 이웃을 샘플링·집계하는 귀납적(inductive) GNN으로, 대규모 그래프에 확장성이 뛰어나다."));
children.push(P("서브그래프 연합학습은 하나의 거대한 그래프가 여러 클라이언트에 부분 그래프(서브그래프)로 나뉘어 분산된 상황을 다룬다. Kim 등(2025)이 ICLR 2025에서 발표한 **FedLoG**(Subgraph Federated Learning for Local Generalization)는 이 환경에서 클라이언트별 지역 과적합과 누락 클래스 문제를 완화하기 위해, 원본이 아닌 **학습 가능한 합성 노드(synthetic node)**를 글로벌하게 공유하고 신뢰 가능한 지식 응축(knowledge condensation) 전략을 사용한다. 본 연구는 FedLoG의 모델 구조(2-계층 GraphSAGE + 합성 노드)와 그 방어 지향 설계를 재현하여 공격 대상으로 삼고, FedLoG가 채택한 데이터 응축·특징 스케일링·분포 노이즈가 그래디언트 유출에 대해 갖는 실제 방어력을 검증하였다."));

// ===========================================================================
// 3. 위협 모델 및 연구 설계
// ===========================================================================
children.push(H1("3. 위협 모델 및 연구 설계"));

children.push(H2("3.1 위협 모델"));
children.push(P("본 연구는 **정직하지만 호기심 많은(honest-but-curious), 나아가 능동적으로 악의적인(malicious)** 서버를 가정한다. 서버는 연합학습 프로토콜을 정상적으로 수행하는 것처럼 보이지만, 내부적으로는 다음 능력을 갖는다."));
children.push(bullet("**그래디언트 단독 탈취:** 다수 클라이언트의 그래디언트를 평균 집계하기 직전, 표적 클라이언트의 그래디언트만 분리하여 확보할 수 있다."));
children.push(bullet("**모델 구조 접근:** 글로벌 모델의 구조와 파라미터를 알고 있다(연합학습에서 모델은 공유되므로 자연스러운 가정이다)."));
children.push(bullet("**적응적 공격(5장):** 방어 기법의 존재와 원리를 파악하고, 방어가 도입한 비밀 변수(예: 스케일 계수)까지 최적화 대상에 포함하는 조인트 최적화를 수행할 수 있다."));
children.push(P("다만 서버는 클라이언트의 **원본 데이터와 비밀 키(예: 비밀 스케일 계수의 실제 값)에는 직접 접근할 수 없다.** 공격의 목표는 오직 가로챈 그래디언트로부터 표적 데이터를 복원하는 것이다."));

children.push(H2("3.2 전체 실험 설계"));
children.push(P("연구 질문에 답하기 위해, 본 연구는 두 개의 실험 트랙을 구성하였다."));
children.push(bullet("**트랙 I (이미지, 4장):** CIFAR-100 이미지를 보유한 3개 클라이언트와 수정 ResNet 모델을 사용하여, 압축·방어 기법별 이미지 복원 오차를 비교한다(RQ1)."));
children.push(bullet("**트랙 II (그래프, 5장):** WikiCS 그래프를 3개 서브그래프로 분할하고 FedLoG 모델로 실제 FedAvg 학습을 수행한 뒤, 노드 특성 복원의 취약성(RQ2)과 방어–적응적 공격–심화 방어의 공방(RQ3)을 검증한다."));
children.push(P("**통계적 엄밀성(본 개정판의 핵심 보완).** 초기 구현은 그래프 트랙의 난수 시드를 고정하지 않은 채 단일 실행(n=1) 결과를 보고하였다. 본 연구는 두 트랙 모두 **서로 다른 5개 시드(n=5)**로 반복하고 평균±표준편차를 보고한다. 트랙 I은 시드마다 다른 표적 이미지를, 트랙 II는 시드마다 다른 그래프 분할과 표적 노드를 사용한다. 절대 복원도는 표본에 따라 크게 변동하므로, 트랙 II에서는 **동일 시드(동일 표적 노드)를 공유하는 시나리오 간 쌍체(paired) 비교**를 병행하여 방어·공격의 순효과를 분리한다. 재현을 위한 드라이버는 experiments/ 디렉터리에 정리하였다."));

children.push(H2("3.3 평가 지표"));
children.push(P("복원 품질은 두 지표로 평가한다. **평균제곱오차(MSE)**는 복원 데이터와 원본의 성분 단위 차이를 측정하며 0에 가까울수록 완벽한 복원을 뜻한다. **코사인 유사도(Cosine Similarity)**는 방향 일치도를 측정하며 1에 가까울수록 형태가 일치한다. 특히 밀집 벡터인 그래프 노드 특성에서 핵심 지표이다(수백만 원소 벡터의 코사인은 단정밀도 누적오차로 1을 초과할 수 있어 본 연구는 배정밀도로 계산하였다)."));
children.push(P("나아가 방어의 **유용성(utility) 비용**을 함께 측정한다. 모델의 노드 분류 정확도와 더불어, 방어가 적용된 그래디언트가 원본 그래디언트와 방향적으로 얼마나 보존되는지를 나타내는 **그래디언트 충실도(gradient fidelity, 두 그래디언트의 코사인 유사도)**를 유용성 프록시로 사용한다. 충실도가 1에 가까울수록 학습 신호 손상이 적어 모델 유용성이 잘 보존됨을 의미한다. 프라이버시(복원도 낮음)와 유용성(충실도 높음)을 함께 봐야 방어를 공정히 평가할 수 있다."));

// ===========================================================================
// 4. 트랙 I
// ===========================================================================
children.push(H1("4. 트랙 I: 이미지 분류 모델에 대한 그래디언트 유출"));

children.push(H2("4.1 대상 모델 설계"));
children.push(P("DLG 공격은 활성화 함수의 미분이 0이 되는 구간이 많으면 그래디언트로부터 입력을 역추적하기 어렵다는 특성이 있다. 이를 고려하여, 본 연구의 대상 모델은 DLG 선행 연구의 설정을 따라 다음과 같이 수정한 ResNet을 사용한다."));
children.push(bullet("**활성화 함수:** ReLU 대신 모든 구간에서 미분이 0이 아닌 **Sigmoid**를 사용하여 그래디언트 역추적을 가능하게 한다."));
children.push(bullet("**해상도 보존:** 모든 합성곱의 stride를 1로 고정하여, 입력 해상도(32×32)가 최종 완전연결층 직전까지 유지되도록 한다."));
children.push(bullet("**모델 규모:** 로컬 검증을 위해 경량 ResNet-8을 기본으로 사용하며, 필요 시 원 논문 규모의 ResNet-56으로 확장 가능하도록 구현하였다."));

children.push(H2("4.2 공격 알고리즘"));
children.push(P("공격은 더미 이미지와 더미 라벨 로짓을 무작위로 초기화한 뒤(픽셀 발산을 막기 위해 0.5 부근에서 시작), Adam 옵티마이저로 300회 반복 최적화한다. 손실 함수는 (1) 더미 그래디언트와 표적 그래디언트의 **코사인 유사도 손실**과 (2) 인접 픽셀 차이를 억제하는 **TV 손실**의 합으로 구성된다. L-BFGS보다 파라미터 민감도가 낮은 Adam을 채택하여 안정성을 확보하였고, 매 반복마다 픽셀 값을 [0, 1] 범위로 클리핑하였다."));

children.push(H2("4.3 방어 기법"));
children.push(P("클라이언트는 그래디언트를 서버로 전송하기 전 다음 방어 기법 중 하나를 적용할 수 있다: 압축 없음(None), 희소화(하위 20/50/80% 제거), 양자화(FP16), 가지치기(전역 하위 20%), Soteria(표현 중요도 기준 상위 20% 마스킹). 각 기법에 대해 동일한 공격을 수행하고 복원 MSE를 측정하였다."));

children.push(H2("4.4 실험 결과 및 분석"));
children.push(P("표 1과 그림 1은 5개 시드(서로 다른 표적 이미지)에 대한 방어 기법별 복원 결과를 평균±표준편차로 정리한 것이다. 먼저 **라벨은 모든 기법·모든 시드에서 100% 복원**되었는데, 이는 분류 손실의 그래디언트 부호로부터 라벨을 분석적으로 얻을 수 있다는 iDLG의 결과와 일치한다."));

children.push(tableCaption("표 1. 이미지(CIFAR-100) 방어 기법별 복원 결과 (멀티시드, n=5)"));
children.push(makeTable(
  [2000, 1600, 2226, 1400, 1800],
  ["방어 기법", "설정", "복원 MSE (mean±std)", "라벨 복원율", "충실도(유용성)"],
  [
    ["없음(None)", "—", "0.0185 ± 0.0121", "100%", "1.000"],
    ["희소화", "하위 20%", "0.0187 ± 0.0118", "100%", "1.000"],
    ["희소화", "하위 50%", "0.0202 ± 0.0124", "100%", "0.999"],
    [{ t: "희소화", bold: true }, { t: "하위 80%", bold: true }, { t: "0.0309 ± 0.0218", bold: true, fill: "DCE9F7" }, { t: "100%", fill: "DCE9F7" }, { t: "0.997", fill: "DCE9F7" }],
    ["양자화", "FP32→FP16", "0.0181 ± 0.0120", "100%", "1.000"],
    ["가지치기", "전역 하위 20%", "0.0190 ± 0.0120", "100%", "1.000"],
    [{ t: "Soteria" }, { t: "표현 상위 20%" }, { t: "0.0186 ± 0.0125" }, { t: "100%" }, { t: "0.731", fill: "FBE3E0" }],
  ],
));

children.push(img(FIG("fig1_image_mse.png")));
children.push(caption("그림 1. 방어 기법별 복원 오차(평균±표준편차, n=5). 80% 희소화만 오차를 소폭 높이며 그마저 분산이 커 효과가 불안정하다."));

children.push(P("결과를 종합하면, **단순한 압축 기법만으로는 그래디언트 유출을 막기 어렵다.** 희소화 20%·50%, 양자화, 전역 가지치기 20%, Soteria 20%의 복원 MSE는 모두 무방어(0.0185±0.0121)와 통계적으로 구분되지 않았다. 하위 80%를 제거하는 극단적 희소화만이 오차를 약 1.7배(0.0309) 높였으나 표준편차(±0.0218)가 커 방어 효과가 불안정하다. 한편 **Soteria(20%)는 그래디언트 충실도를 0.731까지 떨어뜨려 학습 신호를 손상시키면서도 복원 MSE는 무방어와 동일**하여, 유용성만 희생하고 프라이버시 이득은 없는 비효율적 방어임이 드러났다. 이는 RQ1에 대한 답으로, **단순 압축은 실질적 방어가 되지 못하며 정보 훼손형 방어(차분 프라이버시 등)가 요구됨**을 시사한다."));
children.push(P("아울러 초기 보고치(무방어 MSE 0.002168)는 시드 42·단일 이미지(인덱스 15)에서 얻은 한 사례로, 멀티시드 평균(0.0185)보다 한 자릿수 작았다. 이는 단일 실행 보고가 우연히 쉬운 표본에 좌우될 수 있음을 보여주며, 트랙 II에서 더욱 극적으로 나타난다."));

// ===========================================================================
// 5. 트랙 II
// ===========================================================================
children.push(H1("5. 트랙 II: 그래프 신경망에 대한 그래디언트 유출과 공방"));

children.push(H2("5.1 대상 모델: FedLoGModel"));
children.push(P("트랙 II의 대상은 FedLoG의 구조를 재현한 FedLoGModel이다. 이 모델은 2-계층 GraphSAGE 임베더와, 클래스별로 학습되는 합성 노드 파라미터(syn_head, syn_tail)를 결합한다. 노드 임베딩과 합성 노드로부터 계산한 프로토타입 사이의 거리로 로짓을 산출하며, 노드의 연결 차수(degree)에 따라 두 합성 노드 집합의 기여를 시그모이드 가중치로 결합한다. 데이터셋은 Wikipedia 문서 인용 그래프인 **WikiCS**(Mernyei & Cangea, 2020)를 사용하며, 이를 3개 클라이언트의 서브그래프로 분할하였다."));

children.push(H2("5.2 GNN 특화 공격 알고리즘"));
children.push(P("그래프 노드 특성은 이미지처럼 2차원 공간 구조를 갖지 않는 1차원 밀집 벡터이다. 따라서 공간적 매끄러움을 가정하는 TV 손실은 부적합하다. 본 연구의 GNN 공격은 TV 손실을 제거하고, **코사인 유사도 손실과 L2 정규화**를 결합하여 노드 특성 벡터를 역산한다. 표적 노드 한 개의 특성만을 더미 변수로 두고 나머지 노드 특성은 고정한 채, **L-BFGS(strong Wolfe 직선탐색)** 옵티마이저로 최적화하며, 3회의 무작위 재시작(restart)으로 최적해를 탐색한다."));

children.push(H2("5.3 기본 취약성과 결과의 높은 분산"));
children.push(P("먼저 본 연구가 발견한 가장 중요한 사실은, **GNN 노드 특성의 복원도가 표적 노드와 시드에 따라 극심하게 변동한다**는 점이다. 시드를 고정하지 않은 채 동일한 공격을 5회 반복하면 복원 코사인 유사도가 0.518에서 0.999까지(평균 0.85±0.22) 출렁였다. 멀티시드 baseline(미학습 worst-case, n=5)의 복원 코사인은 평균 0.384, 표준편차 0.316으로, 어떤 노드는 거의 완벽히(>0.9) 복원되는 반면 어떤 노드는 거의 복원되지 않았다(<0.1)."));
children.push(calloutBox("주요 관찰(RQ2).", "그래디언트만으로 그래프 노드 특성을 복원하는 위협은 **실재하지만 균일하지 않다.** 어떤 노드는 소수점까지 역산되어 심각한 속성 추론(attribute inference) 위협이 되지만, 평균 복원도와 분산을 함께 보지 않으면 위험을 과대 또는 과소평가하게 된다. 따라서 단일 실행으로 보고된 ‘코사인 0.9999’ 같은 수치는 대표값이 아니며, **다중 시드 평균±표준편차로 보고해야 한다.**", "FDECEA", "C62828"));

children.push(H2("5.4 FedLoG 내장 방어와 그 비신뢰성"));
children.push(P("FedLoG는 프라이버시를 고려해 (1) 합성 노드만 공유하는 데이터 응축, (2) 클라이언트별 비밀 스케일 계수를 곱하는 특징 스케일링, (3) 클래스 분포 노이즈를 채택한다. 본 연구는 이를 재현하고, 절대 복원도의 분산이 크므로 **동일 시드(동일 표적 노드)를 공유하는 쌍체(paired) 비교**로 순효과를 측정하였다."));
children.push(P("그 결과 특징 스케일링의 효과는 **시드 내 평균 Δ코사인 +0.001 ± 0.465**로, 부호가 시드마다 뒤집히는(−0.45 ~ +0.67) 사실상 무효였다. 서버가 비밀 스케일을 모른 채 공격해도 복원도가 신뢰성 있게 낮아지지 않은 것이다. 이는 초기 보고가 ‘방어 적용 시 코사인 0.68’이라는 단일 사례로 ‘효과적’이라 결론지은 것과 배치되며, **특징 스케일링은 신뢰할 만한 방어가 아님**을 보여준다."));

children.push(H2("5.5 적대적 공격: 조인트 최적화와 스케일 불변성"));
children.push(P("적응적 공격자는 특징 스케일링의 존재를 인지하고, **비밀 스케일 계수마저 더미 데이터와 함께 최적화 대상으로 묶어 동시에 추정하는 조인트 최적화(Joint Optimization)** 를 수행한다. 쌍체 비교 결과 적응적 공격은 순진한 공격 대비 **Δ코사인 +0.467 ± 0.414로 5개 시드 전부에서 복원도를 높였다.** 즉 스케일링 방어를 일관되게 무력화한다."));
children.push(P("흥미롭게도 공격이 비밀 스케일을 정확히 맞히지 못해도 복원에 성공했다. 한 시드에서는 실제 스케일 0.49를 2.30으로 1.81이나 오추정했음에도 특성 복원 코사인은 0.99였다. 그 이유는 **복원 품질 지표인 코사인 유사도가 스케일에 불변(scale-invariant)**이기 때문이다. 스케일링은 벡터의 크기만 바꿀 뿐 방향은 보존하므로, 방향만 맞추면 되는 공격자에게 근본적으로 무력하다."));
children.push(calloutBox("중간 결론.", "특징 스케일링 같은 **난독화(obfuscation) 방어는 그래디언트 안에 정보를 그대로 남긴 채 크기만 가린다.** 코사인 기반 공격은 크기에 불변이므로, 가림 변수를 함께 추정하는(혹은 추정할 필요조차 없는) 적응적 공격자 앞에서 근본적으로 파훼된다.", "FFF4E5", "E67E22"));

children.push(H2("5.6 심화 방어: 정보 훼손 기반 메커니즘"));
children.push(P("적응적 공격을 차단하려면 그래디언트의 정보량 자체를 비가역적으로 훼손해야 한다. 본 연구는 두 방어를 적응적 공격과 결합한 최악의 조건에서 검증하였다(쌍체 기준은 적응적 공격 대비)."));
children.push(H3("5.6.1 차분 프라이버시(DP-SGD 양식)"));
children.push(P("초기 구현은 단순히 가우시안 노이즈만 더해 ‘차분 프라이버시’라 불렀으나, 본 연구는 이를 **그래디언트 L2 노름 클리핑(민감도 C 제한) 후 sigma = noise_multiplier·C 의 보정 노이즈를 주입**하는 DP-SGD 양식으로 정정하였다. 그 결과 복원 코사인은 **0.002 ± 0.013**으로, 적응적 공격 대비 **Δ −0.849 ± 0.168(전 시드 음수)** 의 강력하고 일관된 방어를 보였다. 다만 그래디언트 충실도가 0.001로 떨어져 학습 신호가 거의 파괴되므로 **유용성 비용이 가장 크다.**"));
children.push(H3("5.6.2 극단적 희소화(Extreme Sparsification)"));
children.push(P("크기 기준 하위 95%를 0으로 만들고 상위 5%만 전송한다. 복원 코사인은 **0.498 ± 0.320**으로, 적응적 공격 대비 **Δ −0.354 ± 0.223(전 시드 음수)** 의 방어 효과를 보였다. DP보다 방어력은 약하지만 **그래디언트 충실도 0.702를 유지**하여 프라이버시-유용성 균형은 더 우수하다."));

children.push(H2("5.7 결과 종합"));
children.push(P("표 2는 시나리오별 절대 복원도(평균±표준편차)와 유용성을, 표 3은 분산을 제거한 시드 내 쌍체 효과를 정리한 것이다. 그림 2는 절대값(큰 표준편차가 분산을 그대로 보여준다)을, 그림 3은 견고한 쌍체 효과를 시각화한다."));
children.push(tableCaption("표 2. GNN 공방 시나리오별 절대 복원도와 유용성 (미학습, n=5)"));
children.push(makeTable(
  [560, 2466, 2300, 1900, 1800],
  ["단계", "시나리오", "복원 코사인 (mean±std)", "충실도(유용성)", "판정"],
  [
    [{ t: "①" }, "Baseline (방어 없음)", "0.384 ± 0.316", "1.000", { t: "변동 큼" }],
    [{ t: "②" }, "FedLoG (특징 스케일링)", "0.385 ± 0.447", "1.000", { t: "방어 실패", fill: "FBE3E0" }],
    [{ t: "③" }, "Adaptive (조인트 최적화)", { t: "0.852 ± 0.174", bold: true }, "1.000", { t: "공격 우세", fill: "FBE3E0", bold: true }],
    [{ t: "④" }, "DP (클리핑+노이즈)", { t: "0.002 ± 0.013", bold: true }, { t: "0.001", fill: "FFF4E5" }, { t: "방어 성공", fill: "E5F1E6", bold: true }],
    [{ t: "⑤" }, "Sparse (95%)", "0.498 ± 0.320", { t: "0.702", fill: "E5F1E6" }, { t: "방어(약)", fill: "E5F1E6" }],
  ],
  "7A1F1F",
));
children.push(img(FIG("fig2_gnn_defense.png")));
children.push(caption("그림 2. 시나리오별 복원 코사인(평균±표준편차, n=5). 큰 오차막대는 노드 의존적 분산을, DP의 ~0은 일관된 방어를 보여준다."));

children.push(tableCaption("표 3. 시드 내 쌍체 효과 — 분산을 제거한 공방의 순효과 (n=5)"));
children.push(makeTable(
  [3200, 2226, 1800, 1800],
  ["전환 (같은 표적 노드)", "평균 Δ코사인", "시드별 부호", "해석"],
  [
    ["무방어 → 특징 스케일링", "+0.001 ± 0.465", "혼재", { t: "방어 무효" }],
    [{ t: "순진 → 적대적 공격", bold: true }, { t: "+0.467 ± 0.414", bold: true }, { t: "전부 +", fill: "FBE3E0" }, { t: "방어 파훼", fill: "FBE3E0" }],
    [{ t: "적대적 → DP", bold: true }, { t: "−0.849 ± 0.168", bold: true }, { t: "전부 −", fill: "E5F1E6" }, { t: "강력 방어", fill: "E5F1E6" }],
    ["적대적 → 희소화", "−0.354 ± 0.223", { t: "전부 −", fill: "E5F1E6" }, { t: "방어(약)", fill: "E5F1E6" }],
  ],
  "1F4E2E",
));
children.push(img(FIG("fig3_gnn_paired.png")));
children.push(caption("그림 3. 같은 표적 노드 기준 공방 단계별 복원도 변화량 Δ(쌍체, n=5). 적대적 공격은 항상 유출을 높이고(+), DP·희소화는 항상 낮춘다(−)."));

children.push(P("이 결과는 RQ3에 답한다. **난독화 방어(특징 스케일링)는 신뢰할 수 없고(Δ≈0), 적응적 공격에 일관되게 무력화된다(Δ +0.47, 전 시드).** 반면 **정보 훼손형 방어(DP, 극단적 희소화)는 방어 원리가 노출되고 적응적 공격과 결합된 최악의 조건에서도 전 시드에서 일관되게 복원도를 낮춘다(Δ −0.85, −0.35).** 다만 그 방어력은 유용성 비용을 동반하며(DP 충실도 0.001 대 희소화 0.702), 따라서 프라이버시-유용성 절충을 함께 고려해야 한다."));

// ===========================================================================
// 6. 고찰
// ===========================================================================
children.push(H1("6. 고찰"));

children.push(H2("6.1 주요 발견 요약"));
children.push(P("본 연구의 핵심 발견은 네 가지이다. 첫째(방법론), **그래디언트 유출 실험의 결과는 표적 표본·시드에 따라 분산이 매우 크므로 단일 실행(n=1) 보고는 신뢰할 수 없다.** 본 연구가 정정한 초기 수치들(이미지 MSE 0.002→0.0185, 그래프 코사인 0.9999→평균 0.85의 단일 실행은 0.52~0.999로 변동)이 이를 입증한다. 둘째, 통신 효율용 압축 기법(희소화·양자화·가지치기)은 대체로 방어가 되지 못하며, Soteria는 유용성만 희생하고 프라이버시 이득이 없었다. 셋째, **난독화(특징 스케일링)는 코사인 스케일 불변성 때문에 적응적 공격에 일관되게 무력화**되었다. 넷째, **방어의 강건성은 ‘비밀 유지’가 아니라 ‘정보 훼손’에서 나오되, 그 대가로 유용성 비용을 동반**한다."));

children.push(H2("6.2 이미지 도메인과 그래프 도메인의 비교"));
children.push(P("두 도메인 모두 ‘데이터를 직접 전송하지 않는다’는 사실이 안전을 보장하지 않음을 보였으나, 그 양상은 다르다. 이미지는 고차원(3×32×32)이지만 픽셀 간 공간적 상관이 강해 TV 정규화 같은 사전지식이 복원을 돕고, 라벨은 100% 누출되며 복원 MSE는 0.0185±0.012로 표본 간 비교적 안정적이었다. 반면 그래프 노드 특성은 저차원 밀집 벡터로 구조적 제약이 적어, 일부 노드는 거의 완벽히 복원되지만 다른 노드는 거의 복원되지 않는 **극심한 노드 의존적 분산**을 보였다(코사인 0~0.99). 따라서 ‘그래프가 이미지보다 일률적으로 더 위험하다’는 단순 결론은 성립하지 않으며, 위험은 노드의 구조적 위치에 따라 크게 달라진다. 이는 관계형·표 형식 데이터의 프라이버시 위험을 평가할 때 평균뿐 아니라 분산과 최악 사례를 함께 보아야 함을 시사한다."));

children.push(H2("6.3 난독화 방어 대 훼손성 방어"));
children.push(P("5장의 공방은 방어 설계의 근본 원칙을 드러낸다. 특징 스케일링처럼 정보를 ‘가리기만’ 하는 방어는, 가린 방식(비밀 변수)을 공격자가 함께 추정할 수 있으면 무너진다. 이는 케르크호프스의 원리(Kerckhoffs’s principle)—시스템의 안전은 알고리즘의 비밀이 아니라 키의 비밀에 의존해야 한다—를 연합학습 방어에 적용한 것과 같다. 반면 DP 노이즈나 극단적 희소화는 그래디언트가 담은 정보량 자체를 **비가역적으로 감소**시키므로, 공격자가 방어 원리를 완전히 알더라도 잃어버린 정보를 복원할 수 없다. 따라서 실무에서는 난독화에 의존하기보다, 유용성 손실을 감수하더라도 훼손성 방어를 기본값으로 채택해야 한다."));

children.push(H2("6.4 연구의 한계 및 향후 과제"));
children.push(P("초기 구현의 결함(시드 미고정, 단일 클라이언트 학습, 비클리핑 노이즈, 유용성 미측정, n=1 보고)은 본 개정에서 보완하였으나, 다음 한계가 남는다. 첫째, **학습 진행에 따른 유출 추세는 결론에 이르지 못했다.** 라운드 0→20에서 복원도가 평탄했는데(모델 정확도 0.12→0.21로 아직 충분히 학습되지 않음), 단일 40라운드 실행에서 0.14로 낮아진 정황만 관찰되었다. 충분히 수렴한 모델에서의 유출 감소를 엄밀히 확인하는 것은 후속 과제다. 둘째, 두 트랙 모두 배치 크기 1·표적 1개 복원에 집중했고, 그래프 공격은 라벨을 주어진 것으로 가정하였다. 셋째, DP는 클리핑+노이즈로 정정했으나 형식적 프라이버시 예산(ε, δ) 회계는 수행하지 않았다(노이즈 배율로 강도만 제어). 넷째, CIFAR-100 모델은 CPU 제약상 미학습(worst-case) 상태에서 공격하였다. 다섯째, n=5는 분산을 드러내기에 충분하나 좁은 신뢰구간을 위해서는 더 많은 시드와 표적이 바람직하며, 보안 집계·동형암호와의 결합 효과는 다루지 않았다. 향후 연구에서는 ε-회계를 갖춘 DP의 프라이버시-유용성 곡선, 다양한 GNN 구조로의 일반화, 라벨까지 복원하는 배치 단위 공격을 탐구할 필요가 있다."));

// ===========================================================================
// 7. 결론
// ===========================================================================
children.push(H1("7. 결론"));
children.push(P("본 연구는 연합학습의 프라이버시 보장이 ‘원본 데이터를 전송하지 않는다’는 사실만으로는 성립하지 않음을, 이미지와 그래프 두 도메인에 걸친 **재현 가능하고 통계적으로 엄밀한(n=5)** 그래디언트 유출 실험으로 실증하였다. 동시에 본 연구는 방법론적 교훈을 강조한다. 그래디언트 유출의 결과는 표본·시드 의존성이 커서, 단일 실행으로 보고된 인상적인 수치(예: 코사인 0.9999)는 대표값이 아니다. 실제로 동일 코드의 5회 반복은 0.518~0.999로 출렁였으며, 위험은 평균과 분산을 함께 보아야 정확히 평가된다."));
children.push(P("나아가 본 연구는 ‘취약성 → 내장 방어 → 적응적 공격 → 심화 방어’의 공방을, 절대값의 분산을 제거한 **시드 내 쌍체 비교**로 분석하여 견고한 원칙을 도출하였다. 특징 스케일링 같은 난독화 방어는 코사인 유사도의 스케일 불변성 때문에 적응적 공격에 일관되게(전 시드) 무력화된다. 반면 그래디언트 클리핑을 동반한 차분 프라이버시와 극단적 희소화처럼 정보를 비가역적으로 훼손하는 방어만이, 방어 원리가 적에게 노출된 최악의 조건에서도 일관되게 유출을 차단한다. 다만 이러한 훼손형 방어는 그래디언트 충실도로 측정되는 유용성 비용을 동반하므로(DP 0.001 대 희소화 0.702), 프라이버시-유용성 절충이 설계의 핵심 변수가 된다."));
children.push(P("결론적으로, 안전한 연합학습 시스템은 (1) 방어 기법이 공격자에게 알려졌다고 가정하는 보수적 위협 모델 하에서 설계되어야 하고, (2) 차분 프라이버시처럼 정보 훼손을 동반하는 방어를 유용성 절충과 함께 채택해야 하며, (3) 그 효과는 반드시 다중 시드 평균±표준편차로 검증되어야 한다. 이는 프라이버시 보존 기계학습을 실제 서비스에 적용하려는 모든 시도에 적용되는 핵심 교훈이다."));

// ===========================================================================
// 참고문헌
// ===========================================================================
children.push(H1("참고문헌"));
const refs = [
  "Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS), 308–318.",
  "Geiping, J., Bauermeister, H., Dröge, H., & Moeller, M. (2020). Inverting Gradients — How Easy Is It to Break Privacy in Federated Learning? In Advances in Neural Information Processing Systems (NeurIPS) 33.",
  "Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs (GraphSAGE). In Advances in Neural Information Processing Systems (NeurIPS) 30.",
  "Kim, S., Lee, Y., Oh, Y., Lee, N., Yun, S., Lee, J., Kim, S., Yang, C., & Park, C. (2025). Subgraph Federated Learning for Local Generalization (FedLoG). In International Conference on Learning Representations (ICLR 2025), Oral. arXiv:2503.03995.",
  "Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images (CIFAR-10/100). Technical Report, University of Toronto.",
  "McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg). In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), 1273–1282.",
  "Mernyei, P., & Cangea, C. (2020). Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks. arXiv:2007.02901.",
  "Sun, J., Li, A., Wang, B., Yang, H., Li, H., & Chen, Y. (2021). Soteria: Provable Defense Against Privacy Leakage in Federated Learning From Representation Perspective. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9311–9319.",
  "Zhao, B., Mopuri, K. R., & Bilen, H. (2020). iDLG: Improved Deep Leakage from Gradients. arXiv:2001.02610.",
  "Zhu, L., Liu, Z., & Han, S. (2019). Deep Leakage from Gradients. In Advances in Neural Information Processing Systems (NeurIPS) 32.",
];
refs.forEach((r) => children.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 100, line: 300, lineRule: "auto" },
  indent: { left: 360, hanging: 360 },
  children: [new TextRun({ text: r, size: 19 })],
})));

// ===========================================================================
// 문서 조립
// ===========================================================================
const doc = new Document({
  creator: "진로탐색 연구 프로젝트",
  title: "연합학습에서의 그래디언트 유출 공격과 방어의 공방 분석",
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: "1F3864" },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "9DB7D6", space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: "2E5496" },
        paragraph: { spacing: { before: 220, after: 110 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 21, bold: true, font: FONT, color: "44546A" },
        paragraph: { spacing: { before: 160, after: 90 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 600, hanging: 280 } } } }] },
      { reference: "nums", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1)",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 600, hanging: 320 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC", space: 2 } },
        children: [new TextRun({ text: "연합학습 그래디언트 유출 공격과 방어 분석",
          size: 15, color: "999999" })],
      })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({ text: "- ", size: 17, color: "777777" }),
          new TextRun({ children: [PageNumber.CURRENT], size: 17, color: "777777" }),
          new TextRun({ text: " -", size: 17, color: "777777" }),
        ],
      })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  const out = "/Users/hansol/Documents/Study/jinro-tamseok-project/paper/연합학습_그래디언트유출_공격과방어_논문.docx";
  fs.writeFileSync(out, buffer);
  console.log("WROTE:", out);
});
