import {FC} from 'react'
import {Outlet} from 'react-router-dom'
import {useSelector} from 'react-redux'
import {RootState} from '../store'
import {Feedback} from '../entities/Feedback'
import {Affix, Box, Paper} from '@mantine/core'

const MainLayout: FC = () => {
  const feedbacks = useSelector<RootState, Feedback[]>(state => state.application.feedbacks)
  return (
    <Box id="MainLayout">
      <Outlet/>
      <Affix position={{bottom: 20, right: 20}}>
        {feedbacks.map((it, i) => {
          switch (it.level){
            case 'error':
              return (<Paper key={i} c="white" bg="red" p="xs" m="xs">{it.message}</Paper>)
            case 'success':
              return (<Paper key={i} c="white" bg="green" p="xs" m="xs">{it.message}</Paper>)
            case 'warning':
              return (<Paper key={i} c="black" bg="yellow" p="xs" m="xs">{it.message}</Paper>)
            case 'info':
              return (<Paper key={i} c="white" bg="blue" p="xs" m="xs">{it.message}</Paper>)
            default:
              return (<Paper key={i} p="xs" m="xs">{it.message}</Paper>)
          }
        })}
      </Affix>
    </Box>
  )
}

export default MainLayout