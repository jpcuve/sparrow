import Logo from '../logo.svg'
import {FC} from 'react'
import {Center} from '@mantine/core'

const AppView: FC = () => {
  // const delay = (ms: number) => new Promise(fn => setTimeout(fn, ms))
  return (
    <Center>
      <img src={Logo} alt="logo"/>
    </Center>
  )
}

export default AppView