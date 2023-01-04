export interface Visitor {
  accountId: number,
  name: string,
  email: string,
  roles: string[],
  features: string[],
  aspects: string[],
}

export const defaultVisitor: Visitor = {
  accountId: 0,
  name: '',
  email: '',
  roles: [],
  features: [],
  aspects: [],
}

