from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class MarketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    selection: Literal["home", "away"]
    provider: Literal["manual", "polymarket"] = "manual"
    token_id: str = Field(default="", max_length=200, pattern=r"^\d*$")
    mapping_confirmed: bool = False
    bid_price: float | None = Field(default=None, gt=0, lt=1)
    ask_price: float | None = Field(default=None, gt=0, lt=1)
    liquidity: float = Field(default=0, ge=0)
    observed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def check(self):
        if self.provider == "polymarket":
            if not self.token_id or not self.mapping_confirmed:
                raise ValueError("Token ID and confirmation of game/outcome mapping are required")
        elif self.bid_price is None or self.ask_price is None or self.bid_price > self.ask_price:
            raise ValueError("Manual quotes require bid <= ask")
        return self


class PaperRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    signal_id: UUID
    stake: float = Field(gt=0, le=10000)
