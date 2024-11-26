import ctranslate2
from transformers import AutoTokenizer


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Generator, List, Dict
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Request
from protocol import ChatCompletionRequest, CompletionRequest
import time
from protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ErrorResponse,
    FunctionCall,
    RequestResponseMetadata,
    ToolCall,
    UsageInfo,
)

from protocol import (
    CompletionLogProbs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    ErrorResponse,
    RequestResponseMetadata,
    UsageInfo,
)


app = FastAPI()


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

# Load the CTranslate2 model
generator = ctranslate2.Generator(
    "test_ct2", device_index=[0], device="cuda", inter_threads=8
)


# Define a function to generate response text
def generate_response_text(
    input_tokens: list, max_tokens: int, temperature: float, top_p: float
) -> Generator[str, None, None]:
    print(max_tokens)
    # input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))

    outputs = generator.generate_tokens(
        prompt=input_tokens,
        sampling_topp=top_p,
        sampling_temperature=temperature,
        max_length=max_tokens,
        # sampling_topk=1,
    )
    for token in outputs:
        word = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(token.token), skip_special_tokens=True
        )
        print(word)
        yield word


# Define a function to stream the response
def completion_stream_response(
    request_id: str,
    chunk_object_type: str,
    created_time: int,
    model_name: str,
    num_choices: int,
    role: str,
    num_prompt_tokens: int,
    include_continuous_usage: bool,
    tokenizer,
    input_tokens: list,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Generator[str, None, None]:
    completion_tokens = 0
    for word in generate_response_text(input_tokens, max_tokens, temperature, top_p):
        print("word: ", word)
        choice_data = CompletionResponseStreamChoice(
            index=0,
            text=word,
            logprobs=None,
            finish_reason=None,
        )
        chunk = CompletionStreamResponse(
            id=request_id,
            object=chunk_object_type,
            created=created_time,
            choices=[choice_data],
            model=model_name,
        )

        if include_continuous_usage:
            completion_tokens += len(tokenizer.encode(word))
            chunk.usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=num_prompt_tokens + completion_tokens,
            )

        data = chunk.model_dump_json(exclude_unset=True)
        print("here: ", data)
        yield f"data: {data}\n\n"


# Define a function to stream the response
def chat_completion_stream_response(
    request_id: str,
    chunk_object_type: str,
    created_time: int,
    model_name: str,
    num_choices: int,
    role: str,
    num_prompt_tokens: int,
    include_continuous_usage: bool,
    tokenizer,
    input_tokens: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Generator[str, None, None]:
    completion_tokens = 0
    for word in generate_response_text(input_tokens, max_tokens, temperature, top_p):
        print("word: ", word)
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(
                role=role,
                content=word,
            ),
            logprobs=None,
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            object=chunk_object_type,
            created=created_time,
            choices=[choice_data],
            model=model_name,
        )

        if include_continuous_usage:
            completion_tokens += len(tokenizer.encode(word))
            chunk.usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=num_prompt_tokens + completion_tokens,
            )

        data = chunk.model_dump_json(exclude_unset=True)
        print("here: ", data)
        yield f"data: {data}\n\n"


def completion_response(
    request_id: str,
    chunk_object_type: str,
    created_time: int,
    model_name: str,
    num_choices: int,
    role: str,
    num_prompt_tokens: int,
    include_continuous_usage: bool,
    tokenizer,
    input_tokens: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Generator[str, None, None]:
    choices = []
    num_generated_tokens = 0
    results = generator.generate_batch(
        [input_tokens],
        max_length=max_tokens,
        sampling_topk=top_p,
        sampling_temperature=temperature,
        return_scores=True,
    )

    # Get the generated text
    generated_text = tokenizer.decode(
        tokenizer.convert_tokens_to_ids(results[0].sequences_ids[0])
    )

    choice_data = CompletionResponseChoice(
        index=len(choices),
        text=generated_text,
        logprobs=None,
        finish_reason=None,
        stop_reason=None,
        prompt_logprobs=None,
    )
    choices.append(choice_data)
    num_generated_tokens += len(results[0].sequences_ids[0])
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    output = CompletionResponse(
        request_id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )
    return output


@app.post("/v1/completions")
async def chat_completions(request: CompletionRequest, raw_request: Request):
    # try:
    print(request)

    # Tokenize the input
    input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(request.prompt))

    # Generate text
    num_prompt_tokens = len(input_tokens)
    if request.stream == False:
        return completion_response(
            request_id="chatcmpl-123456",
            chunk_object_type="completion.chunk",
            created_time=int(time.time()),
            model_name="ctranslate2-model",
            num_choices=1,
            role="assistant",
            num_prompt_tokens=num_prompt_tokens,
            include_continuous_usage=True,
            tokenizer=tokenizer,
            input_tokens=input_tokens,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    else:
        return StreamingResponse(
            completion_stream_response(
                request_id="chatcmpl-123456",
                chunk_object_type="completion.chunk",
                created_time=int(time.time()),
                model_name="ctranslate2-model",
                num_choices=1,
                role="assistant",
                num_prompt_tokens=num_prompt_tokens,
                include_continuous_usage=True,
                tokenizer=tokenizer,
                input_tokens=input_tokens,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            ),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )


def chat_completion_response(
    request_id: str,
    chunk_object_type: str,
    created_time: int,
    model_name: str,
    num_choices: int,
    role: str,
    num_prompt_tokens: int,
    include_continuous_usage: bool,
    tokenizer,
    input_tokens: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Generator[str, None, None]:
    choices = []
    num_generated_tokens = 0
    results = generator.generate_batch(
        [input_tokens],
        max_length=max_tokens,
        sampling_topk=top_p,
        sampling_temperature=temperature,
        return_scores=True,
    )

    # Get the generated text
    generated_text = tokenizer.decode(
        tokenizer.convert_tokens_to_ids(results[0].sequences_ids[0])
    )

    choice_data = ChatCompletionResponseChoice(
        index=len(choices),
        text=generated_text,
        logprobs=None,
        finish_reason=None,
        stop_reason=None,
        prompt_logprobs=None,
    )
    choices.append(choice_data)
    num_generated_tokens += len(results[0].sequences_ids[0])
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    output = ChatCompletionResponse(
        request_id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )
    return output


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    # try:
    print(request)
    # Concatenate messages into a single prompt
    prompt = tokenizer.apply_chat_template(
        request.messages, add_generation_prompt=True, tokenize=False
    )

    # Tokenize the input
    input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))

    # Generate text

    num_prompt_tokens = len(input_tokens)
    if request.stream == False:
        return chat_completion_response(
            request_id="chatcmpl-123456",
            chunk_object_type="chat.completion.chunk",
            created_time=int(time.time()),
            model_name="ctranslate2-model",
            num_choices=1,
            role="assistant",
            num_prompt_tokens=num_prompt_tokens,
            include_continuous_usage=True,
            tokenizer=tokenizer,
            input_tokens=input_tokens,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    else:
        return StreamingResponse(
            chat_completion_stream_response(
                request_id="chatcmpl-123456",
                chunk_object_type="chat.completion.chunk",
                created_time=int(time.time()),
                model_name="ctranslate2-model",
                num_choices=1,
                role="assistant",
                num_prompt_tokens=num_prompt_tokens,
                include_continuous_usage=True,
                tokenizer=tokenizer,
                input_tokens=input_tokens,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            ),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=6272,
    )
