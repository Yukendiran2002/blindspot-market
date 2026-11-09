// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Liceron
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Annotator } from './Annotator'
import { Console } from './Console'
import './index.css'

const path = location.pathname.replace(/\/+$/, '')
const Page = path === '/label' ? Annotator : Console

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Page />
  </StrictMode>,
)
