import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import Dashboard from "./Dashboard";
document.documentElement.classList.remove('dark');
const el = document.getElementById("root");
if (!el) throw new Error('#root not found');
createRoot(el).render(<React.StrictMode><Dashboard /></React.StrictMode>);