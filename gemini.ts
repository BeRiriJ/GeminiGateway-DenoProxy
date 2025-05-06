import { Buffer } from "node:buffer";

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";
const API_CLIENT = "genai-js/0.21.0";

class HttpError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
  }
}

const fixCors = ({ headers, status, statusText }: { headers?: HeadersInit; status?: number; statusText?: string }) => {
    const headersObj = new Headers(headers);
    headersObj.set("Access-Control-Allow-Origin", "*");
    return { headers: headersObj, status, statusText };
};

const handleOPTIONS = () => {
    return new Response(null, {
        headers: {
            "Access-Control-Allow-Origin": "",
            "Access-Control-Allow-Methods": "",
            "Access-Control-Allow-Headers": "*",
        }
    });
};

const makeHeaders = (apiKey?: string, more?: Record<string, string>) => ({
    "x-goog-api-client": API_CLIENT,
    ...(apiKey && { "x-goog-api-key": apiKey }),
    ...more
});

async function handleModels(apiKey?: string): Promise<Response> {
    const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
        headers: makeHeaders(apiKey),
    });

    let body = response.body;
    if (response.ok) {
        const { models } = await response.json();
        const transformedModels = models.map((model: { name: string }) => ({
            id: model.name.replace("models/", ""),
            object: "model",
            created: 0,
            owned_by: "",
        }));

        const transformedBody = JSON.stringify({
            object: "list",
            data: transformedModels,
        }, null, " ");
        body = transformedBody;
    }

    return new Response(body, fixCors(response));
}


const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";
async function handleEmbeddings(req: any, apiKey?: string): Promise<Response> {
    if (typeof req.model !== "string") {
        throw new HttpError("model is not specified", 400);
    }
    if (!Array.isArray(req.input)) {
        req.input = [req.input];
    }
    let model;
    if (req.model.startsWith("models/")) {
        model = req.model;
    } else {
        req.model = DEFAULT_EMBEDDINGS_MODEL;
        model = `models/${req.model}`;
    }
    const response = await fetch(`${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`, {
        method: "POST",
        headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
        body: JSON.stringify({
            "requests": req.input.map((text: string) => ({
                model,
                content: { parts: { text } },
                outputDimensionality: req.dimensions,
            }))
        })
    });
    let body = response.body;
    if (response.ok) {
      const { embeddings } = await response.json();
        const transformedEmbeddings = embeddings.map(({ values }: { values: number[] }, index: number) => ({
            object: "embedding",
            index,
            embedding: values,
        }));
        const transformedBody = JSON.stringify({
            object: "list",
            data: transformedEmbeddings,
            model: req.model,
        }, null, " ");

        body = transformedBody;
    }

    return new Response(body, fixCors(response));
}

const DEFAULT_MODEL = "gemini-1.5-pro-latest";

async function handleCompletions(req: any, apiKey?: string): Promise<Response> {
    let model = DEFAULT_MODEL;
    switch (true) {
        case typeof req.model !== "string":
            break;
        case req.model.startsWith("models/"):
            model = req.model.substring(7);
            break;
        case req.model.startsWith("gemini-"):
        case req.model.startsWith("learnlm-"):
            model = req.model;
    }

    const TASK = req.stream ? "streamGenerateContent" : "generateContent";
    let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
    if (req.stream) { url += "?alt=sse"; }

    const response = await fetch(url, {
        method: "POST",
        headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
        body: JSON.stringify(await transformRequest(req)),
    });

    let body = response.body;

    if (response.ok) {
        const id = generateChatcmplId();
      if (req.stream) {
          body = response.body
            .pipeThrough(new TextDecoderStream())
            .pipeThrough(new TransformStream({
              transform: parseStream,
              flush: parseStreamFlush,
              buffer: "",
            }))
            .pipeThrough(new TransformStream({
              transform: toOpenAiStream,
              flush: toOpenAiStreamFlush,
              streamIncludeUsage: req.stream_options?.include_usage,
              model, id, last: [],
            }))
            .pipeThrough(new TextEncoderStream());
      } else {
          const text = await response.text();
          body = processCompletionsResponse(JSON.parse(text), model, id);
      }
    }

  return new Response(body, fixCors(response));
}

const harmCategory = [
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
];

const safetySettings = harmCategory.map(category => ({
    category,
    threshold: "BLOCK_NONE",
}));

const fieldsMap: Record<string, string> = {
    stop: "stopSequences",
    n: "candidateCount",
    max_tokens: "maxOutputTokens",
    max_completion_tokens: "maxOutputTokens",
    temperature: "temperature",
    top_p: "topP",
    top_k: "topK",
    frequency_penalty: "frequencyPenalty",
    presence_penalty: "presencePenalty",
};


const transformConfig = (req: any) => {
    let cfg: any = {};
    for (let key in req) {
        const matchedKey = fieldsMap[key];
        if (matchedKey) {
            cfg[matchedKey] = req[key];
        }
    }
    if (req.response_format) {
        switch (req.response_format.type) {
            case "json_schema":
              cfg.responseSchema = req.response_format.json_schema?.schema;
              if (cfg.responseSchema && "enum" in cfg.responseSchema) {
                  cfg.responseMimeType = "text/x.enum";
                  break;
                }
            case "json_object":
                cfg.responseMimeType = "application/json";
                break;
            case "text":
                cfg.responseMimeType = "text/plain";
                break;
            default:
                throw new HttpError("Unsupported response_format.type", 400);
        }
    }
    return cfg;
};

const parseImg = async (url: string) => {
  let mimeType, data;
    if (url.startsWith("http://") || url.startsWith("https://")) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`${response.status} ${response.statusText} (${url})`);
            }
            mimeType = response.headers.get("content-type");
            const buffer = await response.arrayBuffer();
            data = Buffer.from(buffer).toString("base64");
        } catch (err) {
            throw new Error("Error fetching image: " + String(err));
        }
    } else {
      const match = url.match(/^data:(?<mimeType>.?)(;base64)?,(?<data>.+)$/);
        if (!match) {
            throw new Error("Invalid image data: " + url);
        }
        ({ mimeType, data } = match.groups as { mimeType: string; data: string });
    }
    return {
        inlineData: {
            mimeType,
            data,
        },
    };
};

const transformMsg = async ({ role, content }: any) => {
    const parts = [];
    if (!Array.isArray(content)) {
        parts.push({ text: content });
        return { role, parts };
    }

    for (const item of content) {
        switch (item.type) {
            case "text":
                parts.push({ text: item.text });
                break;
            case "image_url":
                parts.push(await parseImg(item.image_url.url));
                break;
          case "input_audio":
                parts.push({
                    inlineData: {
                        mimeType: "audio/" + item.input_audio.format,
                        data: item.input_audio.data,
                    }
                });
              break;
          default:
            throw new TypeError(`Unknown "content" item type: ${item.type}`);
        }
    }
  if (content.every((item: any) => item.type === "image_url")) {
        parts.push({ text: "" });
  }

    return { role, parts };
};

const transformMessages = async (messages: any[]) => {
    if (!messages) { return; }
    const contents = [];
    let system_instruction;
    for (const item of messages) {
        if (item.role === "system") {
            delete item.role;
            system_instruction = await transformMsg(item);
        } else {
            item.role = item.role === "assistant" ? "model" : "user";
            contents.push(await transformMsg(item));
        }
    }
    if (system_instruction && contents.length === 0) {
    contents.push({ role: "model", parts: { text: " " } });
    }
    return { system_instruction, contents };
};

const transformRequest = async (req: any) => ({
    ...await transformMessages(req.messages),
    safetySettings,
    generationConfig: transformConfig(req),
});

const generateChatcmplId = () => {
    const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
    return "chatcmpl-" + Array.from({ length: 29 }, randomChar).join("");
};

const reasonsMap: Record<string, string> = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
};
const SEP = "\n\n|>";

const transformCandidates = (key: string, cand: any) => ({
  index: cand.index || 0,
    [key]: {
        role: "assistant",
        content: cand.content?.parts.map((p: any) => p.text).join(SEP),
    },
    logprobs: null,
    finish_reason: reasonsMap[cand.finishReason] || cand.finishReason,
});

const transformCandidatesMessage = transformCandidates.bind(null, "message");
const transformCandidatesDelta = transformCandidates.bind(null, "delta");

const transformUsage = (data: any) => ({
  completion_tokens: data.candidatesTokenCount,
    prompt_tokens: data.promptTokenCount,
    total_tokens: data.totalTokenCount
});

const processCompletionsResponse = (data: any, model: string, id: string) => {
  return JSON.stringify({
    id,
    choices: data.candidates.map(transformCandidatesMessage),
    created: Math.floor(Date.now()/1000),
    model,
    object: "chat.completion",
    usage: transformUsage(data.usageMetadata),
    });
};

const responseLineRE = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;
async function parseStream(this: { buffer: string }, chunk: string | null, controller: TransformStreamDefaultController) {
  chunk = await chunk;
    if (!chunk) { return; }
    this.buffer += chunk;
    do {
      const match = this.buffer.match(responseLineRE);
      if (!match) { break; }
      controller.enqueue(match[1]);
      this.buffer = this.buffer.substring(match[0].length);
    } while (true);
}
async function parseStreamFlush(this: { buffer: string }, controller: TransformStreamDefaultController) {
  if (this.buffer) {
      console.error("Invalid data:", this.buffer);
      controller.enqueue(this.buffer);
  }
}

function transformResponseStream(this: { id: string; model: string; streamIncludeUsage: boolean; }, data: any, stop: any, first?: any) {
  const item = transformCandidatesDelta(data.candidates[0]);
    if (stop) { item.delta = {}; } else { item.finish_reason = null; }
    if (first) { item.delta.content = ""; } else { delete item.delta.role; }
    const output = {
        id: this.id,
        choices: [item],
        created: Math.floor(Date.now() / 1000),
        model: this.model,
        object: "chat.completion.chunk",
    };
  if (data.usageMetadata && this.streamIncludeUsage) {
    (output as any).usage = stop ? transformUsage(data.usageMetadata) : null;
  }

    return "data: " + JSON.stringify(output) + delimiter;
}

const delimiter = "\n\n";

async function toOpenAiStream(this: { id: string; model: string; streamIncludeUsage: boolean; last: any[] }, chunk: string | null, controller: TransformStreamDefaultController) {
  const transform = transformResponseStream.bind(this);
  const line = await chunk;
  if (!line) { return; }
  let data;
  try {
      data = JSON.parse(line);
  } catch (err) {
      console.error(line);
      console.error(err);
      const length = this.last.length || 1;
      const candidates = Array.from({ length }, (_, index) => ({
        finishReason: "error",
        content: { parts: [{ text: String(err) }] },
          index,
      }));
      data = { candidates };
  }
  const cand = data.candidates[0];
  console.assert(data.candidates.length === 1, "Unexpected candidates count: %d", data.candidates.length);
    cand.index = cand.index || 0;
  if (!this.last[cand.index]) {
        controller.enqueue(transform(data, false, "first"));
  }
    this.last[cand.index] = data;
  if (cand.content) {
    controller.enqueue(transform(data));
  }
}

async function toOpenAiStreamFlush(this: { id: string; model: string; streamIncludeUsage: boolean; last: any[] }, controller: TransformStreamDefaultController) {
  const transform = transformResponseStream.bind(this);
  if (this.last.length > 0) {
    for (const data of this.last) {
      controller.enqueue(transform(data, "stop"));
        }
      controller.enqueue("data: [DONE]" + delimiter);
  }
}

async function handleRequest(request: Request): Promise<Response> {
    const { pathname } = new URL(request.url);
    console.log(`Request URL: ${request.url}`);
    console.log(`Request Pathname: ${pathname}`);

  if (request.method === "OPTIONS") {
    return handleOPTIONS();
    }

  const errHandler = (err: any) => {
        console.error(err);
        return new Response(err.message, fixCors({ status: err.status ?? 500 }));
  };

  try {
    const auth = request.headers.get("Authorization");
    const apiKey = auth?.split(" ")[1];

    const assert = (success: boolean) => {
        if (!success) {
          throw new HttpError("The specified HTTP method is not allowed for the requested resource", 400);
        }
    };


    switch (true) {
      case pathname.endsWith("/chat/completions"):
        assert(request.method === "POST");
            return handleCompletions(await request.json(), apiKey)
                .catch(errHandler);
        case pathname.endsWith("/embeddings"):
        assert(request.method === "POST");
            return handleEmbeddings(await request.json(), apiKey)
                .catch(errHandler);
        case pathname.endsWith("/models"):
        assert(request.method === "GET");
            return handleModels(apiKey)
                .catch(errHandler);
      case pathname === "/":
            return new Response("Proxy is running!", {status: 200})
        default:
            throw new HttpError("404 Not Found", 404);
    }
  } catch (err) {
    return errHandler(err);
  }
}
const port = 8081; // or replace with env variable
Deno.serve({ hostname: "0.0.0.0", port }, handleRequest);
