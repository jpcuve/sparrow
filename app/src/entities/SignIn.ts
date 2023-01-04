export interface SignIn {
  email: string,
  password: string,
}

export const defaultSignIn: SignIn = {
  email: '',
  password: '',
}