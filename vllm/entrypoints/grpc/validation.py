from enum import Enum


class TGISValidationError(str, Enum):
    """This enum holds all TGIS parameter validation failure cases.
    See the equivalent enumeration in TGIS here:
    https://github.ibm.com/ai-foundation/fmaas-inference-server/blob/main/router/src/validation.rs#L238-L271"""
    Temperature = "temperature must be >= 0.05"
    TopP = "top_p must be > 0.0 and <= 1.0"
    TopK = "top_k must be strictly positive"
    TypicalP = "typical_p must be <= 1.0"
    RepetitionPenalty = "repetition_penalty must be > 0.0"
    LengthPenalty = "length_penalty must be >= 1.0 and <= 10.0"
    MaxNewTokens = "max_new_tokens must be <= {0}"
    MinNewTokens = "min_new_tokens must be <= max_new_tokens"
    InputLength = "input tokens ({0}) plus prefix length ({1}) plus min_new_tokens ({2}) must be <= {3}"
    InputLength2 = "input tokens ({0}) plus prefix length ({1}) must be < {2}"
    Tokenizer = "tokenizer error {0}"
    StopSequences = "can specify at most {0} non-empty stop sequences, each not more than {1} UTF8 bytes"
    TokenDetail = "must request input and/or generated tokens to request extra token detail"
    PromptPrefix = "can't retrieve prompt prefix with id '{0}': {1}"
    SampleParametersGreedy = "sampling parameters aren't applicable in greedy decoding mode"

    def error(self, *args, **kwargs):
        """Raises a ValueError with a nicely formatted string"""
        raise ValueError(self.format(*args, **kwargs))
