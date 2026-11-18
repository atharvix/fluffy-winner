import { createContext } from "react";

export type ThemeName = "light-research" | "dark-research" | "high-contrast" | "cyberpunk";

export interface ThemeContextType {
  theme: ThemeName;
  setTheme: (theme: ThemeName) => void;
}

export const THEME_STORAGE_KEY = "qnids_user_theme";

export const ThemeContext = createContext<ThemeContextType | undefined>(undefined);
