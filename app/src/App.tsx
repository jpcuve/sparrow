import React from 'react'
import {BrowserRouter, Routes, Route} from 'react-router-dom'
import MainLayout from './layouts/MainLayout'
import MainView from './views/MainView'
import AppLayout from './layouts/AppLayout'
import AppView from './views/AppView'
import TestView from './views/TestView'
import SubAppLayout from './layouts/SubAppLayout'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout/>}>
          <Route index element={<MainView/>}/>
          <Route path="app" element={<AppLayout items={[
            {value: 'home', label: 'Home'},
            {value: 'developer', label: 'Developer'}
          ]}/>}>
            <Route index element={<AppView/>}/>
            <Route path="home" element={<AppView/>}/>
            <Route path="developer" element={<SubAppLayout items={[
              {value: 'test', label: 'Test'},
            ]}/>}>
              <Route index element={<TestView/>}/>
              <Route path="test" element={<TestView/>}/>
            </Route>
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
