import {FC, useState} from 'react'
import SignInForm from '../forms/SignInForm'
import {SignIn} from '../entities/SignIn'
import {state} from '../store'
import useAuthentication from '../hooks/AuthenticationHook'
import {useNavigate} from 'react-router-dom'
import {Box, Button, Card, Center, Text, Modal} from '@mantine/core'

const MainView: FC = () => {
  const navigate = useNavigate()
  const {signIn, signOut} = useAuthentication()
  const [open, setOpen] = useState<boolean>(false)
  const ok = async (value: SignIn) => {
    try {
      await signIn(value.email, value.password)
      navigate('/app')
      state.notifySuccess('Signed in')
    } catch (e: any) {
      signOut()
      state.notifyError('Bad credentials')
    }
    setOpen(false)
  }
  return (
    <Box h="100vh">
      <Center h="100%">
        <Card shadow="lg">
          <Text size="lg" mb="lg">The Sparrow application</Text>
          <Center>
            <Button type="button" onClick={() => setOpen(true)}>Sign-in</Button>
          </Center>
        </Card>
      </Center>
      <Modal title="Sign-in" opened={open} onClose={() => setOpen(false)} centered>
        <SignInForm onCancel={() => setOpen(false)} onOk={ok}/>
      </Modal>
    </Box>
  )
}

export default MainView