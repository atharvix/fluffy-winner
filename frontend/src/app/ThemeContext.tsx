import React, { useEffect, useState } from "react";
import { ThemeContext, THEME_STORAGE_KEY, type ThemeName } from "./themeContextInstance";

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setThemeState] = useState<ThemeName>(() => {
    const saved = localStorage.getItem(THEME_STORAGE_KEY) as ThemeName | null;
    if (saved && ["light-research", "dark-research", "high-contrast", "cyberpunk"].includes(saved)) {
      return saved;
    }
    // Respect system preference if not set
    if (typeof window !== "undefined" && window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
      return "light-research";
    }
    return "dark-research";
  });

  const setTheme = (newTheme: ThemeName) => {
    setThemeState(newTheme);
    localStorage.setItem(THEME_STORAGE_KEY, newTheme);
  };

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};
