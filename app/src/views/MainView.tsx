import {FC, useState} from 'react'
import SignInForm from '../forms/SignInForm'
import {SignIn} from '../entities/SignIn'
import {state} from '../store'
import api from '../api'
import useAuthentication from '../hooks/AuthenticationHook'
import {useNavigate} from 'react-router-dom'
import {Box, Button, Group, Modal} from '@mantine/core'

const MainView: FC = () => {
  const navigate = useNavigate()
  const {signIn, signOut} = useAuthentication()
  const [open, setOpen] = useState<boolean>(false)
  const ok = async (value: SignIn) => {
    setOpen(false)
    try {
      await signIn(value.email, value.password)
      navigate('/app')
      state.notify({level: 'success', message: 'Signed in'})
    } catch (e: any) {
      signOut()
      state.notify({level: 'error', message: 'Bad credentials'})
    }
  }
  return (
    <Box>
      <Group>
        <Button type="button" onClick={() => setOpen(true)}>App</Button>
        <Button type="button" onClick={() => state.notify({level: 'success', message: 'OKEEE'})}>Snackbar</Button>
      </Group>
      <Modal title="Sign-in" opened={open} onClose={() => setOpen(false)} centered>
        <SignInForm onCancel={() => setOpen(false)} onOk={ok}/>
      </Modal>
    </Box>
  )
}

export default MainView