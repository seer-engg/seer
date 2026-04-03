export type WorkflowCreationMode = 'AUTO_CREATE' | 'ASK_FIRST' | 'ON_ACCEPTANCE';

/**
 * Type of clarification question
 */
export type QuestionType = 'single_choice' | 'multi_choice' | 'resource_picker' | 'account_picker';

/**
 * Account information for account picker questions
 */
export interface AccountPickerAccountInfo {
  id: number;
  display_name: string;
  has_required_scopes: boolean;
  missing_scopes: string[];
}

/**
 * Option for a clarification question
 */
export interface ClarificationQuestionOption {
  value: string;
  label: string;
  is_wildcard?: boolean;
}

/**
 * Clarification question from the agent
 */
export interface ClarificationQuestion {
  question_id: string;
  question: string;
  question_type: QuestionType;
  options: ClarificationQuestionOption[];
  min_selections?: number;
  max_selections?: number | null;
  // Resource picker specific fields
  provider?: string;
  resource_type?: string;
  display_field?: string;
  value_field?: string;
  search_enabled?: boolean;
  hierarchy?: boolean;
  depends_on?: string; // References another question_id
  depends_on_field?: string; // API parameter name (e.g., 'guild_id')
  reasoning?: string;
  // Account picker specific fields
  tool_name?: string; // Tool requiring OAuth (e.g., 'gmail_send_email')
  accounts?: AccountPickerAccountInfo[]; // Available OAuth accounts (fetched dynamically)
  required_scopes?: string[]; // Required scopes for display
}

/**
 * User's answer to a clarification question
 */
export interface ClarificationAnswer {
  question_id: string;
  selected_values: string[];
  custom_input?: string | null;
}

/**
 * Container for multiple clarification questions (new API format)
 */
export interface ClarificationQuestions {
  questions: ClarificationQuestion[];
}

/**
 * Container for multiple clarification answers (new API format)
 */
export interface ClarificationAnswers {
  answers: ClarificationAnswer[];
}
