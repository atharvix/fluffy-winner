import { createContext } from "react";
import { type Language, type TranslationDictionary } from "./translations";

export interface LanguageContextType {
  language: Language;
  lang: Language;
  setLanguage: (lang: Language) => void;
  t: TranslationDictionary;
}

export const LANGUAGE_STORAGE_KEY = "qnids_user_language";

export const LanguageContext = createContext<LanguageContextType | undefined>(undefined);
