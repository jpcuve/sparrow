import {Outlet, useLocation, useNavigate} from 'react-router-dom'
import {FC} from 'react'
import {Box, Flex, Navbar, NavLink, SelectItem} from '@mantine/core'

interface SubAppLayoutProps {
  items: SelectItem[],
}

const SubAppLayout: FC<SubAppLayoutProps> = props => {
  const {items} = props
  const navigate = useNavigate()
  const location = useLocation()
  const selection = location.pathname.split('/')[3] || items[0].value
  return (
    <Flex id="SubAppLayout">
      <Box sx={{flex: '0 1 auto'}}>
        <Navbar>
          <Navbar.Section>
            {items.map(it => (
              <NavLink key={it.value} label={it.label} active={it.value === selection} onClick={() => navigate(it.value)}/>
            ))}
          </Navbar.Section>
        </Navbar>
      </Box>
      <Box p="sm" sx={{flex: '1 1 auto'}}>
        <Outlet/>
      </Box>
    </Flex>
  )
}

export default SubAppLayout