import { useContext } from "react";
import { LanguageContext, type LanguageContextType } from "./languageContextInstance";

export const useTranslation = (): LanguageContextType => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error("useTranslation must be used within a LanguageProvider");
  }
  return context;
};

export const useLanguage = (): LanguageContextType => {
  return useTranslation();
};
