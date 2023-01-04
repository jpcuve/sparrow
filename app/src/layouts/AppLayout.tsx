import {FC, useEffect} from 'react'
import {Outlet, useLocation, useNavigate} from 'react-router-dom'
import useAuthentication from '../hooks/AuthenticationHook'
import api from '../api'
import {state} from '../store'
import {ActionIcon, Box, Flex, Group, Header, MediaQuery, Menu, SelectItem, Tabs} from '@mantine/core'
import {IconMenu, IconUser} from '@tabler/icons'

interface AppLayoutProps {
  items: SelectItem[],
}

const HEADER_HEIGHT = 45

const AppLayout: FC<AppLayoutProps> = props => {
  const {items} = props
  const navigate = useNavigate()
  const location = useLocation()
  const {visitor, signOut} = useAuthentication()
  const selection = location.pathname.split('/')[2] || items[0].value
  const logout = () => {
    signOut()
    navigate('/')
  }
  useEffect(() => {
    (async () => {
      try {
        await api.perpetual()
      } catch (e: any){
        state.notify({level: 'error', message: e.message})
      }
    })()
  }, [])
  return (
    <Box id="AppLayout">
      <Header height={HEADER_HEIGHT}>
        <Flex h="100%" direction="column" justify="center">
          <Group position="apart" p="sm">
            <MediaQuery styles={{display: 'none'}} smallerThan="xs">
              <Tabs value={selection} onTabChange={e => navigate(e !== null ? e : '/')}>
                <Tabs.List>
                  {items.map(it => (
                    <Tabs.Tab key={it.value} value={it.value}>{it.label}</Tabs.Tab>
                  ))}
                </Tabs.List>
              </Tabs>
            </MediaQuery>
            <MediaQuery styles={{display: 'none'}} largerThan="xs">
              <Menu shadow="md">
                <Menu.Target>
                  <ActionIcon>
                    <IconMenu/>
                  </ActionIcon>
                </Menu.Target>
                <Menu.Dropdown>
                  {items.map(it => (
                    <Menu.Item key={it.value} onClick={() => navigate(it.value)}>{it.label}</Menu.Item>
                  ))}
                </Menu.Dropdown>
              </Menu>
            </MediaQuery>
            <Menu shadow="md">
              <Menu.Target>
                <ActionIcon>
                  <IconUser/>
                </ActionIcon>
              </Menu.Target>
              <Menu.Dropdown>
                <Menu.Label>{visitor}</Menu.Label>
                <Menu.Divider/>
                <Menu.Item onClick={logout}>Logout</Menu.Item>
              </Menu.Dropdown>
            </Menu>
          </Group>
        </Flex>
      </Header>
      <Outlet/>
    </Box>
  )
}

export default AppLayout