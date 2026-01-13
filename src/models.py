from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

# --- Enum Definitions ---
class Modality(str, Enum):
    TEXT = "Text"
    IMAGE = "Image"

class FramingType(str, Enum):
    IMPLIES = "IMPLIES"       # 言及・含意
    NORMALIZES = "NORMALIZES" # 当然視
    REINFORCES = "REINFORCES" # 強化

# --- Component Models ---

class Evidence(BaseModel):
    # IDはPython側で生成するためOptionalですが、論理的には必須
    quote: str = Field(..., description="原文からの直接引用")
    source: str = Field(..., description="出典箇所 (例: キャッチコピー, 冒頭ナレーション)")

class DepictedRole(BaseModel):
    name: str = Field(..., description="人物の役割名 (例: 母親, サラリーマン)")
    action: str = Field(..., description="具体的な動作 (例: 料理を運ぶ)")

class TargetAudience(BaseModel):
    name: str = Field(..., description="広告の呼びかけ対象 (例: 主婦層, 就活生)")

class Association(BaseModel):
    name: str = Field(..., description="短い名詞句での連想 (例: 家事＝母の責務)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="確信度 (0.0-1.0)")
    salience: Literal["High", "Medium", "Low"] = "Medium"

class Framing(BaseModel):
    type: FramingType
    concept: str = Field(..., description="強化/当然視されている概念 (例: 性別役割分業)")
    score: float = Field(..., description="判定スコア")
    reason: str = Field(..., description="なぜその判定に至ったかの理由")

# --- Main Logic Models ---

class Expression(BaseModel):
    # IDはPython側で生成
    text: str = Field(..., description="表現単位のテキスト")
    modality: Modality
    evidence: Evidence
    depicted_roles: List[DepictedRole] = []
    targets: List[TargetAudience] = []
    evokes: List[Association] = []
    framing: List[Framing] = []

class Context(BaseModel):
    # IDはPython側で生成
    type: str = Field(..., description="場面の種類 (例: visual_scene, social_setting)")
    desc: str = Field(..., description="文脈の要約")
    source_text: str = Field(..., description="根拠となった描写")

class AdAnalysisResult(BaseModel):
    contexts: List[Context]
    expressions: List[Expression]