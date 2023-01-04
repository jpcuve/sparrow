import {FC} from 'react'
import {defaultSignIn, SignIn} from '../entities/SignIn'
import {Button, Group, PasswordInput, Stack, TextInput} from '@mantine/core'
import {useForm} from '@mantine/form'

interface SignInFormProps {
  onCancel: () => void,
  onOk: (value: SignIn) => void,
}

const SignInForm: FC<SignInFormProps> = props => {
  const {onOk, onCancel} = props
  const form = useForm<SignIn>({
    initialValues: defaultSignIn,
    validate: {
      'email': value => value.length === 0 ? 'Mandatory field' : null,
      'password': value => value.length === 0 ? 'Mandatory field' : null,
    }
  })
  return (
    <form onSubmit={form.onSubmit((values) => onOk(values))} noValidate>
      <Stack>
        <TextInput label="E-mail" placeholder="E-mail" required {...form.getInputProps('email')}/>
        <PasswordInput label="Password" placeholder="Password" required {...form.getInputProps('password')}/>
        <Group>
          <Button w={100} type="submit">OK</Button>
          <Button w={100} variant="outline" onClick={onCancel}>Cancel</Button>
        </Group>
      </Stack>
    </form>
  )
}

export default SignInForm