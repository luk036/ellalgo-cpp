#pragma once

#include <string>

namespace ellalgo {

    /**  Language codes to be used with the EllAlgo class */
    enum class LanguageCode { EN, DE, ES, FR };

    /**
     * @brief A class for saying hello in multiple languages
     */
    class EllAlgo {
        std::string name;

      public:
        /**
         * @brief Creates a new ellalgo
         * @param[in] name the name to greet
         */
        EllAlgo(std::string name);

        /**
         * @brief Creates a localized string containing the greeting
         * @param[in] lang the language to greet in
         * @return a string containing the greeting
         */
        std::string greet(LanguageCode lang = LanguageCode::EN) const;
    };

}  // namespace ellalgo
