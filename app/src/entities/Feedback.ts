export interface Feedback {
  level: 'error'|'success'|'warning'|'info'
  message: string,
}

export const defaultFeedback: Feedback = {level: 'info', message: ''}
