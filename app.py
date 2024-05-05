import sys
import os

import streamlit as st
from organisations_engagees import display_organisations_engagees
from localisation import display_map
from statistiques import main as display_statistics
from ActionsRSE import display_actions_rse
from AnalyseActionsRSE import display_analyse_actions_rse
from partiesprenantes import display_materiality_partiesprenantes

import streamlit as st
from rag_pipeline import rag_pipeline
from data_manager import get_data

# Import modifiédes fonctions liées aux scripts
from projetRSE import display_rse_projects
from labelRSE import display_rse_labels
from entreprises_labellisees import display_labelled_companies
from inspirezvous import *
from collaborons import display_company_selection_for_materiality, display_materiality_matrix
from documentations import display_documentation


def main():

    data = get_data()

    st.markdown(":point_left: Cliquez pour vous inspirer", unsafe_allow_html=True)

    st.sidebar.title("OPEN DATA & IA au service de la RSE")
    section_principale = st.sidebar.radio(
        "Choisissez votre section",
        ["Data Bordeaux métropole", "Data bziiit", "IA RSE", "Documentation"]
    )

    if section_principale == "Data Bordeaux métropole":
        app_mode = st.sidebar.radio(
            "Choisissez votre sous-section",
            ["Localisations", "Organisations engagées", "Statistiques", "Actions RSE", "Analyse actions RSE"]
        )
        if app_mode == "Localisations":
            display_map()
        elif app_mode == "Organisations engagées":
            display_organisations_engagees()
        elif app_mode == "Statistiques":
            display_statistics()
        elif app_mode == "Actions RSE":
            display_actions_rse()
        elif app_mode == "Analyse actions RSE":
            display_analyse_actions_rse()


    elif section_principale == "Data bziiit":
        ia_mode = st.sidebar.radio(
            "Choisissez votre sous-section",
            ["Labels RSE", "Entreprises labellisées", "Fiches entreprises"]
        )
        if ia_mode == "Labels RSE":
            display_rse_labels()
        elif ia_mode == "Entreprises labellisées":
            display_labelled_companies()
        elif ia_mode == "Fiches entreprises":
            data, bziiit_data = fetch_data()
            selected_company = display_company_selection(data)
            display_company_info(data, bziiit_data, selected_company)

    elif section_principale == "IA RSE":
        ia_mode = st.sidebar.radio(
            "Choisissez votre sous-section",
            ["Parties prenantes", "Matrice de matérialité"]
        )
        if ia_mode == "Parties prenantes":
            data, bziiit_data = fetch_data()
            selected_company = display_company_selection_for_materiality(data)
            if selected_company:
                display_materiality_partiesprenantes(selected_company, data, bziiit_data)
        elif ia_mode == "Matrice de matérialité":
            data, bziiit_data = fetch_data()
            selected_company = display_company_selection_for_materiality(data)
            if selected_company:
                display_materiality_matrix(selected_company, data, bziiit_data)


    elif section_principale == "Documentation":
        display_documentation()




    # Instructions communes à toutes les sections
    st.sidebar.markdown("---")

    st.sidebar.title("Recherche des Actions RSE des Entreprises")
    call_for_proposal = st.sidebar.text_area("Entrez votre appel d'offre", "Décrire ici...")

    if st.sidebar.button("Soumettre"):
        st.write(f"Appel d'offre soumis :\n{call_for_proposal}")

        # Recherche des correspondances
        results = rag_pipeline(call_for_proposal)

        # Afficher les résultats
        if results:
            st.write("Voici les entreprises correspondant à votre appel d'offre :")
            for result in results:
                score = round(result[1], 2)
                if score >= 0.6:
                    color = "Green"
                elif score >= 0.3:
                    color = "Orange"
                else:
                    color = "Red"
                st.write(f"Entreprise : {result[0]}, Score : {score} ({color})")
        else:
            st.write("Aucune entreprise correspondante trouvée.")
    st.sidebar.markdown("Powered by **bziiit IA RSE**")
    st.sidebar.markdown("2024 : Open source en Licence MIT")
    st.sidebar.markdown("info@bziiit.com")
    st.sidebar.markdown("---")


if __name__ == "__main__":
    main()