import {FC} from 'react'
import {Outlet} from 'react-router-dom'
import {useSelector} from 'react-redux'
import {RootState} from '../store'
import {Feedback} from '../entities/Feedback'
import {Affix, Box, Notification} from '@mantine/core'

const MainLayout: FC = () => {
  const feedbacks = useSelector<RootState, Feedback[]>(state => state.application.feedbacks)
  return (
    <Box id="MainLayout">
      <Outlet/>
      <Affix position={{bottom: 20, right: 20}}>
        {feedbacks.map((it, i) => {
          switch (it.level){
            case 'error':
              return (<Notification key={i} color="red" title="Error">{it.message}</Notification>)
            case 'success':
              return (<Notification key={i} color="green" title="Success">{it.message}</Notification>)
            case 'warning':
              return (<Notification key={i} color="yellow" title="Warning">{it.message}</Notification>)
            case 'info':
              return (<Notification key={i} color="blue" title="Information">{it.message}</Notification>)
            default:
              return (<Notification key={i}>{it.message}</Notification>)
          }
        })}
      </Affix>
    </Box>
  )
}

export default MainLayout