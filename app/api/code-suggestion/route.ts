import { type NextRequest, NextResponse } from "next/server"
import Groq from "groq-sdk"
import { auth } from "@/auth"

const client = new Groq({ apiKey: process.env.GROQ_API_KEY! })

const MAX_FILE_CHARS = 200_000

interface CodeSuggestionRequest {
  fileContent: string
  cursorLine: number
  cursorColumn: number
  suggestionType: string
  fileName?: string
}

interface CodeContext {
  language: string
  framework: string
  beforeContext: string
  currentLine: string
  afterContext: string
  cursorPosition: { line: number; column: number }
  isInFunction: boolean
  isInClass: boolean
  isAfterComment: boolean
  incompletePatterns: string[]
}

export async function POST(request: NextRequest) {
  const session = await auth()
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
  }

  try {
    const body: CodeSuggestionRequest = await request.json()
    const { fileContent, cursorLine, cursorColumn, suggestionType, fileName } = body

    // Validate input (empty files are allowed)
    if (
      typeof fileContent !== "string" ||
      !Number.isInteger(cursorLine) ||
      !Number.isInteger(cursorColumn) ||
      cursorLine < 0 ||
      cursorColumn < 0 ||
      typeof suggestionType !== "string" ||
      !suggestionType
    ) {
      return NextResponse.json({ error: "Invalid input parameters" }, { status: 400 })
    }

    if (fileContent.length > MAX_FILE_CHARS) {
      return NextResponse.json({ error: "File too large" }, { status: 413 })
    }

    const context = analyzeCodeContext(fileContent, cursorLine, cursorColumn, fileName)
    const prompt = buildPrompt(context, suggestionType)

    let suggestion: string
    try {
      suggestion = await generateSuggestion(prompt)
    } catch (error) {
      console.error("AI generation error:", error)
      return NextResponse.json({ error: "AI service unavailable" }, { status: 502 })
    }

    // Don't echo the user's code back; return only metadata
    return NextResponse.json({
      suggestion,
      metadata: {
        language: context.language,
        framework: context.framework,
        position: context.cursorPosition,
        generatedAt: new Date().toISOString(),
      },
    })
  } catch (error) {
    console.error("Context analysis error:", error)
    return NextResponse.json(
      { error: "Internal server error", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 },
    )
  }
}

/**
 * Analyze the code context around the cursor position
 */
function analyzeCodeContext(content: string, line: number, column: number, fileName?: string): CodeContext {
  const lines = content.split("\n")
  const currentLine = lines[line] || ""

  // 10 lines before and 10 lines after
  const contextRadius = 10
  const startLine = Math.max(0, line - contextRadius)
  const endLine = Math.min(lines.length, line + contextRadius + 1)

  const beforeContext = lines.slice(startLine, line).join("\n")
  const afterContext = lines.slice(line + 1, endLine).join("\n")

  const language = detectLanguage(content, fileName)
  const framework = detectFramework(content)

  return {
    language,
    framework,
    beforeContext,
    currentLine,
    afterContext,
    cursorPosition: { line, column },
    isInFunction: detectInFunction(lines, line),
    isInClass: detectInClass(lines, line),
    isAfterComment: detectAfterComment(currentLine, column),
    incompletePatterns: detectIncompletePatterns(currentLine, column),
  }
}

/**
 * Build AI prompt based on context
 */
function buildPrompt(context: CodeContext, suggestionType: string): string {
  const { currentLine, cursorPosition } = context
  return `Generate a ${suggestionType} suggestion.

Language: ${context.language}
Framework: ${context.framework}

Context:
${context.beforeContext}
${currentLine.substring(0, cursorPosition.column)}|CURSOR|${currentLine.substring(cursorPosition.column)}
${context.afterContext}

Analysis:
- In Function: ${context.isInFunction}
- In Class: ${context.isInClass}
- After Comment: ${context.isAfterComment}
- Incomplete Patterns: ${context.incompletePatterns.join(", ") || "None"}

Provide only the code to insert at |CURSOR|, matching the surrounding indentation and ${context.language} best practices.`
}

/**
 * Generate suggestion using Groq. Throws on failure so the caller can return an error status.
 */
async function generateSuggestion(prompt: string): Promise<string> {
  const completion = await client.chat.completions.create({
    model: "openai/gpt-oss-20b",
    messages: [
      {
        role: "system",
        content: `You are an expert code completion assistant.

Return ONLY the code that should be inserted.
Do not explain anything.
Do not wrap your answer in markdown.
Preserve indentation.
Keep the completion concise.`,
      },
      { role: "user", content: prompt },
    ],
    temperature: 0.3,
    max_tokens: 600, // headroom: reasoning tokens count toward this limit
  })

  let suggestion = completion.choices[0]?.message?.content?.trim() ?? ""

  // Strip markdown fences and the cursor marker if the model echoes them
  suggestion = suggestion.replace(/```[\w]*\n?/g, "").replace(/\|CURSOR\|/g, "").trim()

  return suggestion
}

// ---------- Helper functions for code analysis ----------

function detectLanguage(content: string, fileName?: string): string {
  if (fileName) {
    const ext = fileName.split(".").pop()?.toLowerCase()
    const extMap: Record<string, string> = {
      ts: "TypeScript",
      tsx: "TypeScript",
      js: "JavaScript",
      jsx: "JavaScript",
      mjs: "JavaScript",
      py: "Python",
      java: "Java",
      go: "Go",
      rs: "Rust",
      php: "PHP",
      rb: "Ruby",
      c: "C",
      cpp: "C++",
      cs: "C#",
      css: "CSS",
      html: "HTML",
      json: "JSON",
      vue: "Vue",
    }
    if (ext && extMap[ext]) return extMap[ext]
  }

  // Content-based fallback (no fileName)
  if (/^\s*package\s+\w+/m.test(content) || /^\s*func\s+/m.test(content)) return "Go"
  if (/^\s*def\s+\w+\(.*\):/m.test(content) || /^\s*(from\s+\S+\s+)?import\s+\w+\s*$/m.test(content)) return "Python"
  if (content.includes("interface ") || /:\s*(string|number|boolean)\b/.test(content)) return "TypeScript"

  return "JavaScript"
}

function detectFramework(content: string): string {
  // Check Next.js first: Next files usually import React too
  if (content.includes("next/") || content.includes("getServerSideProps")) return "Next.js"
  if (content.includes("import React") || content.includes("useState")) return "React"
  if (content.includes("import Vue") || content.includes("<template>")) return "Vue"
  if (content.includes("@angular/") || content.includes("@Component")) return "Angular"

  return "None"
}

function detectInFunction(lines: string[], currentLine: number): boolean {
  for (let i = currentLine - 1; i >= 0; i--) {
    const line = lines[i]
    if (!line) continue
    if (/^\s*(export\s+)?(async\s+)?function\b|^\s*def\s|=>\s*\{\s*$|^\s*(async\s+)?\w+\s*\([^)]*\)\s*\{\s*$/.test(line)) return true
    if (/^}/.test(line)) break // closing brace at column 0 ends a top-level block
  }
  return false
}

function detectInClass(lines: string[], currentLine: number): boolean {
  for (let i = currentLine - 1; i >= 0; i--) {
    const line = lines[i]
    if (!line) continue
    if (/^\s*(export\s+)?(abstract\s+)?class\s+/.test(line)) return true
    if (/^}/.test(line)) break
  }
  return false
}

function detectAfterComment(line: string, column: number): boolean {
  const beforeCursor = line.substring(0, column)
  // Ignore "//" in URLs (http://) and "#" that isn't at the start or after whitespace
  return /(^|[^:])\/\/.*$/.test(beforeCursor) || /(^|\s)#.*$/.test(beforeCursor)
}

function detectIncompletePatterns(line: string, column: number): string[] {
  const beforeCursor = line.substring(0, column)
  const trimmed = beforeCursor.trim()
  const patterns: string[] = []

  if (/^(if|while|for)\s*\($/.test(trimmed)) patterns.push("conditional")
  if (/^(function|def)\s*$/.test(trimmed)) patterns.push("function")
  if (/\{\s*$/.test(beforeCursor)) patterns.push("block-or-object")
  if (/\[\s*$/.test(beforeCursor)) patterns.push("array")
  if (/=\s*$/.test(beforeCursor)) patterns.push("assignment")
  if (/\.\s*$/.test(beforeCursor)) patterns.push("method-call")

  return patterns
}
