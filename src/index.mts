import {
  BedrockAgentRuntimeClient,
  RetrieveCommand,
  RetrieveCommandInput,
} from "@aws-sdk/client-bedrock-agent-runtime";
import {
  BedrockRuntimeClient,
  ConverseStreamCommand,
  ConverseStreamCommandInput,
} from "@aws-sdk/client-bedrock-runtime";
import { APIGatewayProxyEvent } from "aws-lambda";

const REGION = process.env.AWS_REGION || "us-east-1";
const KNOWLEDGE_BASE_ID = process.env.KNOWLEDGE_BASE_ID;
const MODEL_ARN = process.env.MODEL_ARN;
const GUARDRAIL_ID = process.env.GUARDRAIL_ID;
const GUARDRAIL_VERSION = process.env.GUARDRAIL_VERSION;

const SYSTEM_PROMPT = `
## SYSTEM ROLE

You are a domain-specific assistant for the CRDC Submission Portal.

The Submission Portal may be referred to by any of the following names:
- CRDC Submission Portal
- CRDC DataHub
- Data Hub
- CRDC-DH
- Any close variants of these names

Your purpose is to answer user questions accurately and concisely about the CRDC Submission Portal.

## KNOWLEDGE BOUNDARIES (STRICT)

- You MUST use ONLY the information provided in <search_results>.
- You MUST NOT use outside knowledge, assumptions, or guesses.
- You MAY provide code snippets if necessitated by the user's question and relevant to the information in <search_results>.

## INPUT STRUCTURE

1) Search results (authoritative source of truth)

<search_results>
$search_results$
</search_results>

2) User question

<question>
$query$
</question>

## RESPONSE RULES

- Answer ONLY the user's question
- Base the answer strictly on <search_results>.
- Do NOT mention or refer to:
  - search results
  - the existence of a search process
  - the tags <search_results> or <question>
  - information being provided to you
- Respond as if the information is inherently known.
- Do NOT guess or infer beyond the provided data.
- Answer in English.
- Keep the response concise and helpful.
- Use bullet points when it improves clarity.
- Use plain ASCII text only.
- Do NOT use markdown or other formatting symbols.
- Use consistent formatting. Do not switch between bullet points and dashes, etc.
- If the question is ambiguous or unclear, ask for clarification instead of guessing.

## RESPONSE FALLBACK

If the answer cannot be determined with certainty from <search_results>, or the question is unrelated to the CRDC Submission Portal, you must respond exactly with:

"I couldn’t find an answer. Please contact support or check the documentation here [link]"

## ADDITIONAL CONTEXT

- The data_models folder contains aggregated Data Model representations in JSON format.
- These models may appear in <search_results> and should be treated as authoritative.
`;

const bedrockAgent = new BedrockAgentRuntimeClient({ region: REGION });
const bedrockRuntime = new BedrockRuntimeClient({ region: REGION });

type InputBody = {
  question: string;
  sessionId: string | null;
  conversationHistory?: Array<{
    role: "user" | "assistant";
    content: string;
  }>;
};

// type Citation = {
//   retrievedReferences?: Array<{
//     content?: { text?: string };
//     location?: { s3Location?: { uri?: string } };
//     metadata?: Record<string, unknown>;
//   }>;
// };

export const handler = awslambda.streamifyResponse(async (event: APIGatewayProxyEvent, responseStream) => {
  try {
    const method = event?.httpMethod || "POST";

    if (method === "OPTIONS") {
      return responseStream.end();
    }

    let body: InputBody;
    if (typeof event.body === "string") {
      try {
        body = JSON.parse(event.body) as InputBody;
      } catch (e: unknown) {
        return responseStream.end(JSON.stringify({ error: "Invalid JSON body", details: (e as Error).message }));
      }
    } else {
      return responseStream.end(JSON.stringify({ error: "Missing request body" }));
    }

    const question = body?.question;
    const sessionId = body?.sessionId || crypto.randomUUID();
    const conversationHistory = body?.conversationHistory || []; // TODO: Use ephemeral storage for history

    if (!question) {
      return responseStream.end(JSON.stringify({ error: "Missing 'question' in request body" }));
    }

    if (!KNOWLEDGE_BASE_ID || !MODEL_ARN) {
      return responseStream.end(
        JSON.stringify({
          error: "Missing environment variables",
          details: {
            KNOWLEDGE_BASE_ID: !!KNOWLEDGE_BASE_ID,
            MODEL_ARN: !!MODEL_ARN,
          },
        })
      );
    }

    // Step 1: Retrieve relevant documents from Knowledge Base
    const retrieveParams: RetrieveCommandInput = {
      knowledgeBaseId: KNOWLEDGE_BASE_ID,
      retrievalQuery: {
        text: question,
      },
      retrievalConfiguration: {
        vectorSearchConfiguration: {
          numberOfResults: 15,
        },
      },
    };

    let searchResults = "";
    // let citations: Citation = {};

    try {
      const retrieveCommand = new RetrieveCommand(retrieveParams);
      const retrieveResponse = await bedrockAgent.send(retrieveCommand);

      // Build search results context from retrieved documents
      if (retrieveResponse.retrievalResults) {
        // citations = {
        //   retrievedReferences: retrieveResponse.retrievalResults.map((result) => ({
        //     content: { text: result.content?.text },
        //     location: result.location,
        //     metadata: result.metadata,
        //   })),
        // };

        searchResults = retrieveResponse.retrievalResults
          .map((result, index) => {
            const content = result.content?.text || "";
            const source = result.location?.s3Location?.uri || "Unknown source";
            return `[Document ${index + 1}]\nSource: ${source}\nContent: ${content}\n`;
          })
          .join("\n");
      }
    } catch (retrieveError) {
      console.error("Knowledge Base retrieval error:", retrieveError);
      return responseStream.end(
        JSON.stringify({ error: "Failed to retrieve knowledge", details: (retrieveError as Error).message })
      );
    }

    // Step 2: Build conversation messages with context
    const userMessageWithContext = `Search Results:
${searchResults}

User Question: ${question}`;

    const messages = [
      ...conversationHistory.map((msg) => ({
        role: msg.role,
        content: [{ text: msg.content }],
      })),
      {
        role: "user" as const,
        content: [{ text: userMessageWithContext }],
      },
    ];

    // Step 3: Call Converse API
    const converseParams: ConverseStreamCommandInput = {
      modelId: MODEL_ARN.split("/").pop() || MODEL_ARN,
      messages: messages,
      system: [{ text: SYSTEM_PROMPT }],
      inferenceConfig: {
        temperature: 0.2,
        maxTokens: 4096,
      },
      guardrailConfig: {
        guardrailIdentifier: GUARDRAIL_ID,
        guardrailVersion: GUARDRAIL_VERSION,
      },
    };

    try {
      const converseCommand = new ConverseStreamCommand(converseParams);
      const converseResponse = await bedrockRuntime.send(converseCommand);

      // Send session ID first
      responseStream.write(JSON.stringify({ sessionId }) + "\n");

      // Stream the response
      if (converseResponse.stream) {
        for await (const chunk of converseResponse.stream) {
          if (chunk.contentBlockDelta?.delta?.text) {
            responseStream.write(
              JSON.stringify({
                output: chunk.contentBlockDelta.delta.text,
                citation: {}, // citations,
              }) + "\n"
            );
          }

          if (chunk.messageStop) {
            // Stream complete
            break;
          }
        }
      }

      responseStream.end();
    } catch (converseError) {
      console.error("Converse API error:", converseError);
      return responseStream.end(
        JSON.stringify({ error: "Failed to generate response", details: (converseError as Error).message })
      );
    }
  } catch (err: unknown) {
    console.error("Handler error:", err);

    /* @ts-expect-error untyped error */
    return responseStream.end(JSON.stringify({ error: err.message || "Internal server error" }));
  }
});
