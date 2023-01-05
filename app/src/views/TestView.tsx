import {FC} from 'react'
import {Button, Stack} from '@mantine/core'
import {state} from '../store'
import api from '../api'

const TestView: FC = () => {
  const handleClick = async () => {
    try {
      await api.longProcess()
    } catch (e: any){
      state.notifyError(e.message)
    }
  }
  return (
    <Stack>
      <Button onClick={() => state.notifyWarning('Hello')}>Open snackbar</Button>
      <Button onClick={handleClick}>Test</Button>
    </Stack>
  )
}

export default TestView