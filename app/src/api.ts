import remote from "./remote"
import {state} from './store'
import {Perpetual} from './entities/Perpetual'
import {Option} from './entities/Option'
import {Marker} from './entities/Marker'

async function wrap<T>(fn: () => Promise<T>): Promise<T> {
  state.updateFetching(true)
  try {
    return await fn()
  } catch (e: any) {
    state.notify({level: 'error', message: e.message}, 5000)
    throw e
  } finally {
    state.updateFetching(false)
  }
}

// const getVisitor: () => Visitor = () => JSON.parse(localStorage.getItem(`visitor_${process.env.REACT_APP_CLIENT_ID}`) || JSON.stringify(defaultVisitor)) as Visitor

class Api {
  private api = `${process.env.REACT_APP_REMOTE_URL}/web`

/*
  async token(search: URLSearchParams) { return remote.post<any>(`${this.identity}/token`, search) }
  async userinfo() { return remote.get<any>(`${this.identity}/userinfo`) }
*/
  async status() { return remote.get<any>(`${this.api}/`) }
  async perpetual() { state.updatePerpetual(await wrap(() => remote.get<Perpetual>(`${this.api}/perpetual`))) }
  async markers(type: string){ return remote.get<Marker[]>(`${this.api}/markers/${type}`)}
  async childOptions(type: string, parent: string[]){ return remote.post<Option[]>(`${this.api}/child-options/${type}`, parent)}
}

export default new Api()