import api from '../api'
import {useState} from 'react'

const useAuthentication = () => {
  const visitorKey = `visitor_${process.env.REACT_APP_CLIENT_ID}`
  const tokenKey = `token_${process.env.REACT_APP_CLIENT_ID}`
  const [visitor] = useState<string>(localStorage.getItem(visitorKey) || '')
  const [data] = useState<any>({
    signIn: async (username: string, password: string) => {
      console.log('Coucou')
      const token = btoa(`${username}:${password}`);
      console.log(`Token: ${token}`)
      localStorage.setItem(tokenKey, token)
      try{
        await api.status()
        console.log(`Logged in user: ${username}`)
        localStorage.setItem(visitorKey, username)
        return username
      } catch (e: any){
        console.log(`Failed to get user info`)
      }
    },
    signOut: () => {
      localStorage.removeItem(tokenKey)
      localStorage.removeItem(visitorKey)
      return ''
    },
    token: localStorage.getItem(tokenKey) || '',
    signedIn: visitor.length > 0,
  })
  return {
    ...data,
    visitor,
  }
}

export default useAuthentication