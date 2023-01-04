import {configureStore, createSlice, PayloadAction} from '@reduxjs/toolkit'
import {Feedback} from './entities/Feedback'
import {defaultPerpetual, Perpetual} from './entities/Perpetual'

export interface ApplicationState {
  fetching: boolean,
  feedbacks: Feedback[],
  perpetual: Perpetual,
}

const defaultApplicationState: ApplicationState = {
  fetching: false,
  feedbacks: [],
  perpetual: defaultPerpetual,
}

const applicationSlice = createSlice({
  name: 'application',
  initialState: defaultApplicationState,
  reducers: {
    updateFetching(state: ApplicationState, action: PayloadAction<boolean>){
      return {...state, fetching: action.payload}
    },
    updateFeedbacks(state: ApplicationState, action: PayloadAction<Feedback[]>){
      return {...state, feedbacks: action.payload}
    },
    updatePerpetual(state: ApplicationState, action: PayloadAction<Perpetual>){
      return {...state, perpetual: action.payload}
    },
  }
})

// next functions (type: string, payload: any): only create actions
const {updateFetching, updateFeedbacks, updatePerpetual} = applicationSlice.actions


export const store = configureStore({
  reducer:{
    application: applicationSlice.reducer
  }
})

export type RootState = ReturnType<typeof store.getState>

export const state = {
  updateFetching: (fetching: boolean) => store.dispatch(updateFetching(fetching)),
  updatePerpetual: (perpetual: Perpetual) => store.dispatch(updatePerpetual(perpetual)),
  notify: (feedback: Feedback, timeout: number = 2000) => {
    const fs = store.getState().application.feedbacks
    store.dispatch(updateFeedbacks([...fs, feedback]))
    setTimeout(() => {
      const fs = store.getState().application.feedbacks
      store.dispatch(updateFeedbacks(fs.filter(it => it !== feedback)))
    }, timeout)
  },
}
